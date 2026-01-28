import argparse
import os
import sys
import numpy as np
import torch
import hydra
from src.utils.hydra import init_config
from src.data import Data, InstanceData
from src.transforms import instantiate_transforms
import warnings

# 忽略一些不必要的警告
warnings.filterwarnings("ignore")

# 将项目根目录添加到路径
sys.path.append(os.getcwd())

def read_cloud(path):
    """
    读取点云文件
    参数:
        path: 文件路径 (.txt, .npy, .ply)
    返回:
        Data 对象，包含 pos 和 rgb
    """
    print(f"正在读取文件: {path}")
    ext = os.path.splitext(path)[1].lower()
    
    if ext == '.txt':
        # 假设 S3DIS 格式: x y z r g b
        try:
            data = np.loadtxt(path)
        except Exception as e:
            print(f"读取 txt 文件出错: {e}")
            sys.exit(1)
            
        if data.shape[1] < 3:
            raise ValueError("文件必须至少包含 3 列 (x, y, z)")
            
        pos = data[:, :3]
        if data.shape[1] >= 6:
            rgb = data[:, 3:6]
        else:
            rgb = np.zeros_like(pos) # 如果没有颜色，默认为黑色
            
        return Data(pos=torch.from_numpy(pos).float(), rgb=torch.from_numpy(rgb).float())
        
    elif ext == '.npy':
        data = np.load(path)
        pos = data[:, :3]
        if data.shape[1] >= 6:
            rgb = data[:, 3:6]
        else:
            rgb = np.zeros_like(pos)
        return Data(pos=torch.from_numpy(pos).float(), rgb=torch.from_numpy(rgb).float())
        
    elif ext == '.ply':
        try:
            from plyfile import PlyData
        except ImportError:
            print("请安装 plyfile 以读取 .ply 文件: pip install plyfile")
            sys.exit(1)
            
        plydata = PlyData.read(path)
        vertex = plydata['vertex']
        
        x = vertex['x']
        y = vertex['y']
        z = vertex['z']
        pos = np.stack([x, y, z], axis=1)
        
        if 'red' in vertex and 'green' in vertex and 'blue' in vertex:
            r = vertex['red']
            g = vertex['green']
            b = vertex['blue']
            rgb = np.stack([r, g, b], axis=1)
        else:
            rgb = np.zeros_like(pos)
            
        return Data(pos=torch.from_numpy(pos).float(), rgb=torch.from_numpy(rgb).float())
        
    else:
        raise ValueError(f"不支持的文件扩展名: {ext}")

def inference(ply_path, ckpt_path, output_path, conf_threshold=0.5):
    """
    执行 SPT 推理并生成掩码
    参数:
        ply_path: 输入点云路径
        ckpt_path: 模型权重路径
        output_path: 输出掩码路径 (.pt)
        conf_threshold: 置信度阈值
    """
    # 1. 加载配置
    # 设置 PROJECT_ROOT 环境变量用于 Hydra 配置解析
    os.environ["PROJECT_ROOT"] = os.path.dirname(os.path.abspath(__file__))
    
    # 初始化配置，覆盖部分参数以适应推理模式
    cfg = init_config(config_name="train.yaml", overrides=[
        "experiment=panoptic/s3dis_with_stuff",
        "datamodule.dataloader.batch_size=1",
        "datamodule.dataloader.num_workers=0",
        f"ckpt_path={ckpt_path}"
    ])
    
    # 2. 实例化 Datamodule 以获取变换 (Transforms)
    print("正在实例化 Datamodule (获取变换)...")
    
    # 获取预处理变换 (pre_transform)
    pre_transform_cfg = cfg.datamodule.get('pre_transform')
    if pre_transform_cfg is not None:
        print("实例化 pre_transform...")
        pre_transform = instantiate_transforms(pre_transform_cfg)
    else:
        pre_transform = None
        
    # 获取设备上的变换 (on_device_transform)
    # 通过实例化 datamodule 获取是最安全的方式
    datamodule = hydra.utils.instantiate(cfg.datamodule)
    on_device_transform = datamodule.on_device_val_transform
    
    # 3. 实例化模型
    print("正在实例化模型...")
    model = hydra.utils.instantiate(cfg.model)
    
    # 4. 加载权重
    print(f"正在从 {ckpt_path} 加载权重...")
    checkpoint = torch.load(ckpt_path, map_location=lambda storage, loc: storage)
    model.load_state_dict(checkpoint['state_dict'])
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.eval().to(device)
    
    # 5. 读取输入
    nag = read_cloud(ply_path)
    N_raw = nag.num_nodes
    print(f"原始点云点数: {N_raw}")
    
    # 6. 应用变换
    print("正在应用变换...")
    if pre_transform is not None:
        nag = pre_transform(nag)
        
    nag = nag.to(device)
    
    if on_device_transform is not None:
        nag = on_device_transform(nag)
        
    # 兼容 Data 和 NAG 对象的属性访问
    if hasattr(nag, 'num_nodes'):
        print(f"变换后节点数: {nag.num_nodes}, 边数: {nag.num_edges}")
    else:
        # NAG 对象，访问 Level-0
        print(f"变换后 NAG 节点数 (Level-0): {nag[0].num_nodes}, 边数: {nag[0].num_edges}")

    # 7. 推理
    print("正在运行推理...")
    with torch.no_grad():
        output = model(nag)
        
    # 8. 处理输出
    print("正在处理输出...")
    
    # 调试信息：打印图结构相关的属性
    if hasattr(output, 'obj_edge_index') and output.obj_edge_index is not None:
        print(f"DEBUG: output.obj_edge_index shape: {output.obj_edge_index.shape}")
    else:
        print("DEBUG: output.obj_edge_index is None")

    # 获取超点映射关系
    # nag[0].super_index: Level-0 (Voxel) -> Level-1 (Superpoint)
    super_index_level0_to_level1 = nag[0].super_index
    
    # 构建 Raw -> Level-0 的映射
    # 因为 OpenIns3D 需要原始点云上的掩码，而 SPT 可能会对点云进行体素化下采样 (GridSampling)
    # nag[0].sub 包含了从原始点到 Level-0 的映射信息
    
    if hasattr(nag[0], 'sub'):
        sub = nag[0].sub
        if hasattr(sub, 'points') and hasattr(sub, 'sizes'):
            # Cluster 对象情况
            # sub.points: 原始点的索引
            # sub.sizes: 每个组 (Level-0 节点) 的大小
            # 我们需要反转这个关系得到: raw_point_idx -> level0_node_idx
            
            sub_cpu = sub.cpu()
            super_index_raw_to_level0 = torch.full((N_raw,), -1, dtype=torch.long, device=device)
            
            cluster_idx = torch.arange(sub_cpu.num_groups, device=device)
            indices = sub_cpu.points.to(device)
            values = cluster_idx.repeat_interleave(sub_cpu.sizes.to(device).long())
            
            # 边界检查
            if indices.max() >= N_raw:
                print(f"Warning: indices max {indices.max()} >= N_raw {N_raw}. Clipping.")
                mask = indices < N_raw
                indices = indices[mask]
                values = values[mask]
                
            super_index_raw_to_level0[indices] = values
        else:
            # 可能是 Tensor
            super_index_raw_to_level0 = sub.to(device)
    else:
        # 如果没有下采样 (不太可能，通常都有 GridSampling)
        # 假设一对一映射
        print("Warning: No sub-sampling info found. Assuming 1-to-1 mapping.")
        super_index_raw_to_level0 = torch.arange(N_raw, device=device)

    # 获取 Level-0 的实例预测
    # voxel_panoptic_pred 返回: (semantic_pred, instance_index, instance_data)
    # 这是最基础的预测，不依赖复杂的图边计算，相对健壮
    _, vox_index, _ = output.voxel_panoptic_pred(super_index=super_index_level0_to_level1)
    
    # 计算实例数量
    if vox_index.numel() > 0:
        n_instances = int(vox_index.max().item()) + 1
    else:
        n_instances = 0
        
    print(f"Level-0 发现实例数: {n_instances}")
    
    if n_instances == 0:
        mask_tensor = torch.zeros((N_raw, 0), dtype=torch.bool, device=device)
    else:
        # 尝试获取实例置信度分数进行过滤
        # 这一步可能会因为图结构异常（如只有1个节点导致没有边）而失败
        # 如果失败，我们回退到保留所有实例
        try:
            # panoptic_pred 计算基于图的复杂分数 (Inter/Intra affinity)
            obj_score, _, _ = output.panoptic_pred()
            
            # 过滤
            keep = torch.where(obj_score > conf_threshold)[0]
            print(f"基于置信度 {conf_threshold} 过滤后保留实例数: {len(keep)}")
            
        except Exception as e:
            print(f"WARNING: panoptic_pred() 计算失败: {e}")
            print("原因分析: 这通常是因为输入点云在体素化后形成的图结构过于简单（例如只有一个超点或没有边），导致无法计算对象间的亲和度分数。")
            print("处理方案: 跳过置信度过滤，保留所有预测出的实例。")
            
            # 回退：保留所有实例
            keep = torch.arange(n_instances, device=device)
        
        if keep.numel() == 0:
            mask_tensor = torch.zeros((N_raw, 0), dtype=torch.bool, device=device)
        else:
            # 构建掩码矩阵: [N_raw, N_kept]
            # 1. 先构建 Level-0 的掩码: [N_grid, N_kept]
            # vox_index: [N_grid]
            # keep: [N_kept]
            
            # 向量化生成掩码
            # masks_l0[i, j] = True if vox_index[i] == keep[j]
            masks_l0 = vox_index.unsqueeze(1) == keep.unsqueeze(0)
            
            # 2. 映射回原始点: [N_raw, N_kept]
            valid_mask = super_index_raw_to_level0 >= 0
            mapped_indices = super_index_raw_to_level0[valid_mask]
            
            mask_tensor = torch.zeros((N_raw, masks_l0.shape[1]), dtype=torch.bool, device=device)
            
            # 将 Level-0 的掩码赋值给对应的原始点
            mask_tensor[valid_mask] = masks_l0[mapped_indices]
            
    print(f"最终生成掩码形状: {mask_tensor.shape}")
    
    # 保存结果
    torch.save(mask_tensor.cpu(), output_path)
    print(f"结果已保存至 {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ply_path", type=str, required=True, help="输入点云文件路径")
    parser.add_argument("--ckpt_path", type=str, required=True, help="SPT 模型权重路径")
    parser.add_argument("--output_path", type=str, required=True, help="输出掩码保存路径")
    parser.add_argument("--conf_threshold", type=float, default=0.5, help="置信度阈值")
    
    args = parser.parse_args()
    
    inference(args.ply_path, args.ckpt_path, args.output_path, args.conf_threshold)
