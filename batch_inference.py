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


def save_predictions(path: str, nag, dataset_name: str = None):
    """
    保存预测结果到文件。
    
    Args:
        path: 输出文件路径
        nag: 包含预测结果的 NAG 对象
        dataset_name: 数据集名称（用于格式调整）
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    data = nag[0]
    pos = data.pos.cpu().numpy()
    rgb = data.rgb.cpu().numpy() if data.rgb is not None else np.zeros_like(pos)
    
    if rgb.max() <= 1.0:
        rgb = rgb * 255
    
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
    
    sp_columns = [sp.reshape(-1, 1) for sp in sp_indices]
    if sp_columns:
        output_data = np.column_stack([pos, rgb, pred, obj_pred] + sp_columns)
    else:
        output_data = np.column_stack([pos, rgb, pred, obj_pred])
    
    fmt_list = ['%.6f', '%.6f', '%.6f', '%d', '%d', '%d', '%d', '%d']
    fmt_list.extend(['%d'] * len(sp_indices))
    fmt = ' '.join(fmt_list)
    
    ext = os.path.splitext(path)[1].lower()
    
    if ext == '.ply':
        header_lines = [
            "ply",
            "format ascii 1.0",
            f"element vertex {output_data.shape[0]}",
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
        header_lines.append("end_header")
        header = "\n".join(header_lines)
        np.savetxt(path, output_data, fmt=fmt, header=header, comments='')
    else:
        np.savetxt(path, output_data, fmt=fmt)
    
    print(f"Saved: {path}")


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
    
    print("Loading configuration...")
    cfg = init_config(config_name="train.yaml", overrides=[
        f"experiment={args.experiment}",
        "datamodule.dataloader.batch_size=1",
        "datamodule.dataloader.num_workers=0",
        f"ckpt_path={args.ckpt}"
    ])
    
    print("Instantiating datamodule...")
    datamodule = hydra.utils.instantiate(cfg.datamodule)
    datamodule.setup(stage=args.stage)
    
    print("Instantiating model...")
    model = hydra.utils.instantiate(cfg.model)
    
    on_device_transform = getattr(datamodule, 'on_device_test_transform', None)
    if on_device_transform is None:
        on_device_transform = getattr(datamodule, 'on_device_val_transform', None)
    
    print(f"Loading checkpoint from {args.ckpt}...")
    checkpoint = torch.load(args.ckpt, map_location='cpu')
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
    
    for idx in tqdm(scene_indices, desc="Processing scenes"):
        try:
            batch = dataset[idx]
            
            # NAG 对象的点数据在 batch[0].pos 中，而不是 batch.pos
            if hasattr(batch, 'num_levels'):
                # NAG 对象
                if not hasattr(batch[0], 'pos') or batch[0].pos is None or batch[0].pos.shape[0] == 0:
                    print(f"Skipping empty scene at index {idx}")
                    continue
            else:
                # Data 对象
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
            
            if on_device_transform:
                batch = on_device_transform(batch)
                
            batch = run_inference_on_batch(model, batch, device)
            
            output_file = os.path.join(args.output_dir, f"{scene_id_safe}_pred.{args.output_format}")
            save_predictions(output_file, batch, args.experiment)
            
            if args.visualize:
                output_html = os.path.join(args.output_dir, f"{scene_id_safe}_vis.html")
                generate_visualization(
                    batch, output_html, class_names, class_colors,
                    stuff_classes, num_classes
                )
            
            num_points = batch[0].pos.shape[0]
            results_summary.append({
                'scene_id': scene_id,
                'num_points': num_points,
                'output_file': output_file
            })
            
        except Exception as e:
            print(f"Error processing scene {idx}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    summary_file = os.path.join(args.output_dir, "inference_summary.txt")
    with open(summary_file, 'w') as f:
        f.write(f"Experiment: {args.experiment}\n")
        f.write(f"Checkpoint: {args.ckpt}\n")
        f.write(f"Stage: {args.stage}\n")
        f.write(f"Total scenes processed: {len(results_summary)}\n")
        f.write("-" * 50 + "\n")
        for result in results_summary:
            f.write(f"{result['scene_id']}: {result['num_points']} points -> {result['output_file']}\n")
    
    print(f"\nInference complete!")
    print(f"Total scenes processed: {len(results_summary)}")
    print(f"Results saved to: {args.output_dir}")
    print(f"Summary saved to: {summary_file}")


if __name__ == "__main__":
    main()
