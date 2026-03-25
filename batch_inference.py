"""
Batch inference script for entire dataset.
Supports S3DIS, KITTI-360, and ScanNet datasets.
Generates predictions and optional visualizations for all scenes in the dataset.

Usage:
    python batch_inference.py \
        --experiment panoptic/s3dis \
        --ckpt path/to/checkpoint.ckpt \
        --output_dir results/s3dis_predictions \
        --stage test \
        --visualize
"""

import argparse
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, Future
from tqdm import tqdm
import numpy as np
import torch
import hydra
from omegaconf import DictConfig

sys.path.append(os.getcwd())

from src.utils.hydra import init_config
from src.data import Data, InstanceData
from src.transforms import instantiate_transforms
from src.visualization.visualization import visualize_3d


def get_dataset_config(experiment: str):
    """
    根据实验名称获取数据集配置。
    
    Args:
        experiment: 实验名称，如 'panoptic/s3dis', 'panoptic/kitti360', 'panoptic/scannet'
    
    Returns:
        tuple: (class_names, class_colors, num_classes, stuff_classes)
    """
    dataset_name = experiment.split('/')[-1].lower()
    
    if 's3dis' in dataset_name:
        from src.datasets.s3dis_config import (
            CLASS_NAMES, CLASS_COLORS, S3DIS_NUM_CLASSES, 
            STUFF_CLASSES, STUFF_CLASSES_MODIFIED
        )
        class_names = CLASS_NAMES
        class_colors = CLASS_COLORS
        num_classes = S3DIS_NUM_CLASSES
        stuff_classes = STUFF_CLASSES_MODIFIED if 'stuff' in dataset_name else STUFF_CLASSES
        
    elif 'kitti' in dataset_name:
        from src.datasets.kitti360_config import (
            CLASS_NAMES, CLASS_COLORS, KITTI360_NUM_CLASSES, STUFF_CLASSES
        )
        class_names = CLASS_NAMES
        class_colors = CLASS_COLORS
        num_classes = KITTI360_NUM_CLASSES
        stuff_classes = STUFF_CLASSES
        
    elif 'scannet' in dataset_name:
        from src.datasets.scannet_config import (
            CLASS_NAMES, CLASS_COLORS, SCANNET_NUM_CLASSES, STUFF_CLASSES
        )
        class_names = CLASS_NAMES
        class_colors = CLASS_COLORS
        num_classes = SCANNET_NUM_CLASSES
        stuff_classes = STUFF_CLASSES
        
    else:
        print(f"Warning: Unknown dataset in experiment '{experiment}'")
        class_names, class_colors, num_classes, stuff_classes = None, None, None, None
    
    return class_names, class_colors, num_classes, stuff_classes


def _prepare_save_data(nag):
    """
    从 NAG 对象中提取并准备需要保存的数据（numpy 数组）。
    在 GPU 推理完成后调用，将所有 tensor 移到 CPU 并转为 numpy。

    Args:
        nag: 包含预测结果的 NAG 对象

    Returns:
        dict: 包含 pos, rgb, pred, obj_pred, sp_indices, has_hspt_hard, is_hard 的字典
    """
    data = nag[0]
    pos = data.pos.cpu().numpy()

    if hasattr(data, 'pos_offset') and data.pos_offset is not None:
        pos_offset = data.pos_offset.cpu().numpy()
        pos = pos + pos_offset

    if hasattr(data, 'rgb') and data.rgb is not None:
        rgb = data.rgb.cpu().numpy()
        if rgb.max() <= 1.0:
            rgb = rgb * 255
    else:
        rgb = np.zeros_like(pos)

    if data.semantic_pred is not None:
        pred = data.semantic_pred.cpu().numpy()
        if pred.ndim > 1:
            pred = np.argmax(pred, axis=1)
    else:
        pred = np.zeros(pos.shape[0], dtype=np.int32)

    obj_pred = np.full(pos.shape[0], -1, dtype=np.int32)

    if hasattr(data, 'obj_pred') and data.obj_pred is not None:
        if isinstance(data.obj_pred, InstanceData):
            try:
                indices = data.obj_pred.indices.cpu().numpy()
                obj_ids = data.obj_pred.values[0].cpu().numpy()
                obj_pred[indices] = obj_ids
            except Exception as e:
                print(f"Warning: Error processing InstanceData: {e}")
        elif isinstance(data.obj_pred, torch.Tensor):
            obj_pred = data.obj_pred.cpu().numpy()
        elif isinstance(data.obj_pred, np.ndarray):
            obj_pred = data.obj_pred

    sp_indices = []
    if hasattr(nag, 'num_levels'):
        num_levels = nag.num_levels
        current_sp = None
        for i in range(num_levels - 1):
            if hasattr(nag[i], 'super_index') and nag[i].super_index is not None:
                if i == 0:
                    current_sp = nag[i].super_index.cpu().numpy()
                else:
                    current_sp = nag[i].super_index.cpu().numpy()[current_sp]
                sp_indices.append(current_sp)

    has_hspt_hard = hasattr(data, 'is_hspt_hard')
    is_hard = None
    if has_hspt_hard:
        is_hard = data.is_hspt_hard.cpu().numpy().astype(np.int32)

    return {
        'pos': pos, 'rgb': rgb, 'pred': pred, 'obj_pred': obj_pred,
        'sp_indices': sp_indices, 'has_hspt_hard': has_hspt_hard, 'is_hard': is_hard,
        'num_points': pos.shape[0]
    }


def save_predictions(path: str, save_data: dict, dataset_name: str = None):
    """
    保存预测结果到文件。使用二进制 PLY 格式代替 ASCII 以大幅提升写入速度。
    对于 152 万点的场景，二进制写入比 ASCII 快 10-100 倍。

    Args:
        path: 输出文件路径
        save_data: 由 _prepare_save_data 返回的字典
        dataset_name: 数据集名称（用于格式调整）
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)

    pos = save_data['pos']
    rgb = save_data['rgb']
    pred = save_data['pred']
    obj_pred = save_data['obj_pred']
    sp_indices = save_data['sp_indices']
    has_hspt_hard = save_data['has_hspt_hard']
    is_hard = save_data['is_hard']
    num_points = save_data['num_points']

    ext = os.path.splitext(path)[1].lower()

    if ext == '.ply':
        _save_binary_ply(path, pos, rgb, pred, obj_pred, sp_indices,
                         has_hspt_hard, is_hard, num_points)
    else:
        # txt 格式退化为快速的 numpy 二进制保存
        sp_columns = [sp.reshape(-1, 1) for sp in sp_indices]
        cols = [pos, rgb, pred.reshape(-1, 1), obj_pred.reshape(-1, 1)] + sp_columns
        if has_hspt_hard:
            cols.append(is_hard.reshape(-1, 1))
        output_data = np.column_stack(cols)
        fmt_list = ['%.6f'] * 3 + ['%d'] * 3 + ['%d'] * 2
        fmt_list.extend(['%d'] * len(sp_indices))
        if has_hspt_hard:
            fmt_list.append('%d')
        np.savetxt(path, output_data, fmt=' '.join(fmt_list))

    print(f"Saved: {path}")


def _save_binary_ply(path, pos, rgb, pred, obj_pred, sp_indices,
                     has_hspt_hard, is_hard, num_points):
    """
    以二进制 little-endian 格式保存 PLY 文件。
    相比 ASCII 格式，写入速度提升 10-100 倍，文件体积缩小约 75%。

    Args:
        path: 输出文件路径
        pos: 点坐标 (N, 3) float
        rgb: 颜色 (N, 3) uint8
        pred: 语义预测 (N,) int
        obj_pred: 实例预测 (N,) int
        sp_indices: 超点索引列表
        has_hspt_hard: 是否有 HSPT 困难标记
        is_hard: HSPT 困难标记 (N,) int 或 None
        num_points: 点数
    """
    # 构建 PLY header
    header_lines = [
        "ply",
        "format binary_little_endian 1.0",
        f"element vertex {num_points}",
        "property float x",
        "property float y",
        "property float z",
        "property uchar red",
        "property uchar green",
        "property uchar blue",
        "property int semantic_pred",
        "property int obj_pred"
    ]
    for i in range(len(sp_indices)):
        header_lines.append(f"property int sp_level_{i+1}")
    if has_hspt_hard:
        header_lines.append("property int is_hspt_hard")
    header_lines.append("end_header")
    header = "\n".join(header_lines) + "\n"

    # 使用 numpy structured array 向量化写入，避免 Python 逐点循环瓶颈
    # 构建 dtype：3 float32 + 3 uint8 + N int32
    dt_fields = [
        ('x', '<f4'), ('y', '<f4'), ('z', '<f4'),
        ('r', 'u1'), ('g', 'u1'), ('b', 'u1'),
        ('semantic_pred', '<i4'), ('obj_pred', '<i4'),
    ]
    for i in range(len(sp_indices)):
        dt_fields.append((f'sp_level_{i+1}', '<i4'))
    if has_hspt_hard:
        dt_fields.append(('is_hspt_hard', '<i4'))

    vertex_dtype = np.dtype(dt_fields)
    vertices = np.empty(num_points, dtype=vertex_dtype)

    # 向量化赋值（无 Python 循环）
    vertices['x'] = pos[:, 0].astype(np.float32)
    vertices['y'] = pos[:, 1].astype(np.float32)
    vertices['z'] = pos[:, 2].astype(np.float32)
    rgb_u8 = np.clip(rgb, 0, 255).astype(np.uint8)
    vertices['r'] = rgb_u8[:, 0]
    vertices['g'] = rgb_u8[:, 1]
    vertices['b'] = rgb_u8[:, 2]
    vertices['semantic_pred'] = pred.astype(np.int32)
    vertices['obj_pred'] = obj_pred.astype(np.int32)
    for i, sp in enumerate(sp_indices):
        vertices[f'sp_level_{i+1}'] = sp.astype(np.int32)
    if has_hspt_hard and is_hard is not None:
        vertices['is_hspt_hard'] = is_hard.astype(np.int32)

    with open(path, 'wb') as f:
        f.write(header.encode('ascii'))
        f.write(vertices.tobytes())


def generate_visualization(nag, output_html: str, class_names, class_colors, 
                           stuff_classes, num_classes):
    """
    生成可视化 HTML 文件。
    
    Args:
        nag: NAG 对象
        output_html: 输出 HTML 路径
        class_names: 类别名称列表
        class_colors: 类别颜色列表
        stuff_classes: stuff 类别索引列表
        num_classes: 类别数量
    """
    nag = nag.to('cpu')
    
    vis_output = visualize_3d(
        nag,
        class_names=class_names,
        class_colors=class_colors,
        stuff_classes=stuff_classes,
        num_classes=num_classes,
        max_points=100000,
        centroids=True,
        h_edge=True,
        h_edge_width=2
    )
    
    fig = vis_output['figure']
    fig.update_layout(
        autosize=True,
        width=None,
        height=None,
        margin=dict(l=0, r=0, b=0, t=0)
    )
    
    fig.write_html(output_html, config={'responsive': True})
    print(f"Visualization saved: {output_html}")


def run_inference_on_batch(model, batch, device):
    """
    对单个 batch 执行推理。
    
    Args:
        model: 模型
        batch: 输入数据 batch
        device: 计算设备
    
    Returns:
        NAG: 包含预测结果的 NAG 对象
    """
    batch = batch.to(device)
    
    # Ensure super_index and other indices are long to prevent torch_scatter from crashing
    if hasattr(batch, 'num_levels'):
        for i in range(getattr(batch, 'num_levels')):
            if hasattr(batch[i], 'super_index') and batch[i].super_index is not None:
                batch[i].super_index = batch[i].super_index.long()
            if hasattr(batch[i], 'edge_index') and batch[i].edge_index is not None:
                batch[i].edge_index = batch[i].edge_index.long()

    with torch.no_grad():
        output = model(batch)
    
    batch[0].semantic_pred = output.voxel_semantic_pred(super_index=batch[0].super_index)
    
    try:
        res = output.voxel_panoptic_pred(super_index=batch[0].super_index)
        if isinstance(res, tuple) and len(res) == 3:
            _, _, vox_obj_pred = res
            batch[0].obj_pred = vox_obj_pred
    except Exception as e:
        pass
    
    # 抽取 HSPT 困难超点标签
    if hasattr(output, 'hspt_output') and output.hspt_output is not None:
        if hasattr(batch, 'num_levels') and batch.num_levels >= 2:
            super_idx = batch[0].super_index
            hard_idx_l1 = output.hspt_output.hard_sp_indices
            mask = torch.isin(super_idx, hard_idx_l1)
            batch[0].is_hspt_hard = mask
    
    return batch


def main():
    parser = argparse.ArgumentParser(description="Batch inference on entire dataset")
    parser.add_argument('--experiment', type=str, required=True,
                        help='Experiment config name (e.g., panoptic/s3dis, panoptic/kitti360)')
    parser.add_argument('--ckpt', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--output_dir', type=str, default='output/batch_predictions',
                        help='Output directory for predictions')
    parser.add_argument('--stage', type=str, default='test', 
                        choices=['train', 'val', 'test', 'trainval'],
                        help='Dataset stage to run inference on')
    parser.add_argument('--visualize', action='store_true',
                        help='Generate visualization HTML for each scene')
    parser.add_argument('--output_format', type=str, default='txt',
                        choices=['txt', 'ply'],
                        help='Output file format')
    parser.add_argument('--limit', type=int, default=None,
                        help='Limit number of scenes to process (for testing)')
    parser.add_argument('--scene_ids', type=str, nargs='+', default=None,
                        help='Specific scene IDs to process (e.g., Area_5/office_1)')
    args = parser.parse_args()
    
    os.environ["PROJECT_ROOT"] = os.path.dirname(os.path.abspath(__file__))
    
    print(f"Experiment: {args.experiment}")
    print(f"Checkpoint: {args.ckpt}")
    print(f"Output directory: {args.output_dir}")
    print(f"Stage: {args.stage}")
    
    class_names, class_colors, num_classes, stuff_classes = get_dataset_config(args.experiment)
    
    print(f"Loading checkpoint from {args.ckpt}...")
    checkpoint = torch.load(args.ckpt, map_location='cpu')
    hspt_enabled = any('hspt.' in k for k in checkpoint['state_dict'].keys())
    
    print("Loading configuration...")
    overrides = [
        f"experiment={args.experiment}",
        "datamodule.dataloader.batch_size=1",
        "datamodule.dataloader.num_workers=0",
        f"ckpt_path={args.ckpt}"
    ]
    if hspt_enabled:
        print("Detected H-SPT weights, enabling model.hspt.enable=True")
        overrides.append("model.hspt.enable=True")
        
        # Auto-detect HSPT d_raw from checkpoint
        d_raw = 6
        for k, v in checkpoint['state_dict'].items():
            if 'hspt.cafm.point_encoder.0.weight' in k:
                d_raw = v.shape[1]
                break
                
        if d_raw == 3:
            print("Auto-detected HSPT raw_keys=['pos'] from checkpoint")
            overrides.append("model.hspt.raw_keys=[pos]")
            overrides.append("model.hspt.d_raw=3")
        elif d_raw == 6:
            print("Auto-detected HSPT raw_keys=['pos','rgb'] from checkpoint")
            overrides.append("model.hspt.raw_keys=[pos,rgb]")
            overrides.append("model.hspt.d_raw=6")
        
    cfg = init_config(config_name="train.yaml", overrides=overrides)
    
    print("Instantiating datamodule...")
    datamodule = hydra.utils.instantiate(cfg.datamodule)
    
    # Force loading 'rgb' if it's missing (e.g., in KITTI-360 config which uses HSV)
    if hasattr(datamodule, 'kwargs') and 'point_load_keys' in datamodule.kwargs:
        if datamodule.kwargs['point_load_keys'] is not None and 'rgb' not in datamodule.kwargs['point_load_keys']:
            datamodule.kwargs['point_load_keys'] = list(datamodule.kwargs['point_load_keys']) + ['rgb']
            
    datamodule.setup(stage=args.stage)
    
    print("Instantiating model...")
    model = hydra.utils.instantiate(cfg.model)
    
    on_device_transform = getattr(datamodule, 'on_device_test_transform', None)
    if on_device_transform is None:
        on_device_transform = getattr(datamodule, 'on_device_val_transform', None)
    
    model.load_state_dict(checkpoint['state_dict'])
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model = model.eval().to(device)
    
    if hasattr(model, 'net'):
        model.net.store_features = True
    
    if args.stage == 'train':
        dataset = datamodule.train_dataset
    elif args.stage == 'val':
        dataset = datamodule.val_dataset
    elif args.stage == 'trainval':
        train_dataset = datamodule.train_dataset
        val_dataset = datamodule.val_dataset
        from torch.utils.data import ConcatDataset
        dataset = ConcatDataset([train_dataset, val_dataset])
    else:
        dataset = datamodule.test_dataset
    
    cloud_ids = dataset.cloud_ids if hasattr(dataset, 'cloud_ids') else None
    
    if args.scene_ids:
        if cloud_ids is not None:
            scene_id_to_idx = {cid: idx for idx, cid in enumerate(cloud_ids)}
            valid_scene_ids = [sid for sid in args.scene_ids if sid in scene_id_to_idx]
            if not valid_scene_ids:
                print(f"Warning: None of the specified scene IDs found.")
                print(f"Available IDs (first 10): {cloud_ids[:10]}...")
            scene_indices = [scene_id_to_idx[sid] for sid in valid_scene_ids]
        else:
            scene_indices = [int(sid) for sid in args.scene_ids if sid.isdigit()]
    else:
        scene_indices = list(range(len(dataset)))
    
    if args.limit:
        scene_indices = scene_indices[:args.limit]
    
    print(f"Processing {len(scene_indices)} scenes...")
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    results_summary = []
    
    # --- 优化: 使用线程池异步保存/可视化，以及后台预加载 ---
    io_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix='io_worker')
    io_futures: list[Future] = []
    
    def _preload_scene(ds, scene_idx):
        """
        在后台线程中预加载下一个场景的数据。
        
        Args:
            ds: 数据集对象
            scene_idx: 场景索引
        
        Returns:
            预加载的 batch 数据
        """
        return ds[scene_idx]
    
    preload_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='preload')
    prefetch_future: Future = None
    
    total_infer_time = 0.0
    total_io_time = 0.0
    
    for i, idx in enumerate(tqdm(scene_indices, desc="Processing scenes")):
        try:
            # --- 获取数据: 优先使用预加载的结果 ---
            t_load_start = time.perf_counter()
            if prefetch_future is not None:
                batch = prefetch_future.result()
            else:
                batch = dataset[idx]
            t_load_end = time.perf_counter()
            
            # --- 提交下一个场景的预加载 ---
            if i + 1 < len(scene_indices):
                next_idx = scene_indices[i + 1]
                prefetch_future = preload_executor.submit(_preload_scene, dataset, next_idx)
            else:
                prefetch_future = None
            
            # NAG 对象的点数据在 batch[0].pos 中，而不是 batch.pos
            if hasattr(batch, 'num_levels'):
                if not hasattr(batch[0], 'pos') or batch[0].pos is None or batch[0].pos.shape[0] == 0:
                    print(f"Skipping empty scene at index {idx}")
                    continue
            else:
                if not hasattr(batch, 'pos') or batch.pos is None or batch.pos.shape[0] == 0:
                    print(f"Skipping empty scene at index {idx}")
                    continue
            
            scene_id = None
            if cloud_ids is not None and idx < len(cloud_ids):
                scene_id = cloud_ids[idx]
            elif hasattr(batch, 'cloud_id'):
                scene_id = batch.cloud_id
            elif hasattr(batch, 'id'):
                scene_id = batch.id
            else:
                scene_id = f"scene_{idx}"
            
            scene_id_safe = scene_id.replace('/', '_').replace('\\', '_')
            
            # --- GPU 推理 ---
            if on_device_transform:
                batch = on_device_transform(batch)
            
            t_infer_start = time.perf_counter()
            batch = run_inference_on_batch(model, batch, device)
            torch.cuda.synchronize()  # 确保 GPU 推理完成后再计时
            t_infer_end = time.perf_counter()
            total_infer_time += (t_infer_end - t_infer_start)
            
            # --- 在提交IO前，先在主线程中提取数据到 CPU numpy ---
            num_points = batch[0].pos.shape[0]
            save_data = _prepare_save_data(batch)
            
            # --- 异步提交保存和可视化到线程池 ---
            output_file = os.path.join(args.output_dir, f"{scene_id_safe}_pred.{args.output_format}")
            io_futures.append(
                io_executor.submit(save_predictions, output_file, save_data, args.experiment)
            )
            
            if args.visualize:
                output_html = os.path.join(args.output_dir, f"{scene_id_safe}_vis.html")
                # 可视化需要完整的 nag 对象，先移到 CPU
                batch_cpu = batch.to('cpu')
                io_futures.append(
                    io_executor.submit(
                        generate_visualization, batch_cpu, output_html,
                        class_names, class_colors, stuff_classes, num_classes
                    )
                )
            
            results_summary.append({
                'scene_id': scene_id,
                'num_points': num_points,
                'output_file': output_file
            })
            
            print(f"  [{scene_id}] load={t_load_end - t_load_start:.1f}s, "
                  f"infer={t_infer_end - t_infer_start:.1f}s, points={num_points}")
            
        except Exception as e:
            print(f"Error processing scene {idx}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # --- 等待所有后台 IO 任务完成 ---
    print("Waiting for async IO tasks to complete...")
    for f in io_futures:
        try:
            f.result()
        except Exception as e:
            print(f"IO task error: {e}")
    
    io_executor.shutdown(wait=True)
    preload_executor.shutdown(wait=True)
    
    summary_file = os.path.join(args.output_dir, "inference_summary.txt")
    with open(summary_file, 'w') as f:
        f.write(f"Experiment: {args.experiment}\n")
        f.write(f"Checkpoint: {args.ckpt}\n")
        f.write(f"Stage: {args.stage}\n")
        f.write(f"Total scenes processed: {len(results_summary)}\n")
        f.write(f"Total inference time: {total_infer_time:.2f}s\n")
        f.write("-" * 50 + "\n")
        for result in results_summary:
            f.write(f"{result['scene_id']}: {result['num_points']} points -> {result['output_file']}\n")
    
    print(f"\nInference complete!")
    print(f"Total scenes processed: {len(results_summary)}")
    print(f"Total GPU inference time: {total_infer_time:.2f}s")
    print(f"Results saved to: {args.output_dir}")
    print(f"Summary saved to: {summary_file}")


if __name__ == "__main__":
    main()
