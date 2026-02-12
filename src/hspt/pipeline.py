"""
H-SPT Pipeline 工具函数

提供训练/推理流水线所需的工具函数：
- build_packed_points: 从 NAG 构造 packed 点张量
- run_hspt: 三模块串联 + refine loss 计算
- HSPTOutput: 输出数据结构
"""

import warnings
from dataclasses import dataclass
from typing import Optional, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from .aus import AdaptiveUncertaintySampler
from .cafm import CrossAttentionFusionModule
from .rrh import ResidualRefinementHead


__all__ = [
    'build_packed_points',
    'run_hspt',
    'HSPTOutput',
    'HSPTModule',
]


@dataclass
class HSPTOutput:
    """
    H-SPT 模块输出数据结构。
    
    属性:
        fused_features: [M, D] 融合后的全局超点特征（用于主分类头）
        hard_sp_indices: [K] 困难超点索引
        point_logits: [K, N, C] 点级分类 logits（用于 refine loss）
        packed_point_idx: [K, N] 采样点在原始点云中的索引
        packed_mask: [K, N] 有效点掩码
    """
    fused_features: torch.Tensor
    hard_sp_indices: torch.Tensor
    point_logits: torch.Tensor
    packed_point_idx: Optional[torch.Tensor] = None
    packed_mask: Optional[torch.Tensor] = None


def build_packed_points(
    nag,
    n_sample: int = 64,
    raw_keys: List[str] = None,
    device: torch.device = None
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    从 NAG 构造 packed 点张量（向量化实现）。
    
    参数:
        nag: NAG 对象，包含多层级点云数据
        n_sample: 每个超点采样的点数，默认 64
        raw_keys: 原始点属性键列表，默认 ['pos']
        device: 目标设备
    
    返回:
        packed_raw_points: [M, N, d_raw] 打包的原始点张量
        packed_point_idx: [M, N] 采样点索引
        packed_mask: [M, N] 有效点掩码 (True=有效, False=padding)
    """
    if raw_keys is None:
        raw_keys = ['pos']
    
    # 获取设备
    if device is None:
        device = nag[0].pos.device
    
    # 获取超点数量
    M = nag[1].num_nodes if hasattr(nag[1], 'num_nodes') else nag[1].pos.shape[0]
    
    # 边界处理：M=0
    if M == 0:
        d_raw = sum(getattr(nag[0], k).shape[-1] if hasattr(nag[0], k) and getattr(nag[0], k) is not None else 0 for k in raw_keys)
        d_raw = max(d_raw, 3)
        return (
            torch.zeros(0, n_sample, d_raw, device=device),
            torch.full((0, n_sample), -1, dtype=torch.long, device=device),
            torch.zeros(0, n_sample, dtype=torch.bool, device=device)
        )
    
    # 获取点->超点映射
    super_index = nag[0].super_index if hasattr(nag[0], 'super_index') else None
    N_atoms = nag[0].pos.shape[0]
    
    if super_index is None:
        warnings.warn("NAG 缺少 super_index，使用随机采样作为回退", RuntimeWarning)
        packed_point_idx = torch.randint(0, max(N_atoms, 1), (M, n_sample), device=device)
        packed_mask = torch.ones(M, n_sample, dtype=torch.bool, device=device)
    else:
        # ===========================
        # 向量化实现（避免 Python for 循环）
        # ===========================
        super_index = super_index.to(device)
        
        # 统计每个超点的点数
        counts = torch.bincount(super_index, minlength=M).to(device)
        
        # 对每个点分配一个随机排名（用于采样）
        rand_perm = torch.rand(N_atoms, device=device)
        
        # 按超点分组排序，获取每个超点内的局部排名
        sorted_rand, sorted_indices = rand_perm.sort()
        sorted_super = super_index[sorted_indices]
        
        # 重新按 super_index 排序
        order = super_index.argsort(stable=True)
        
        # 计算每个点在其超点内的局部索引
        ptr = torch.cat([torch.zeros(1, dtype=torch.long, device=device), counts.cumsum(0)])
        
        # 使用 scatter 计算每个点的局部排名
        local_rank = torch.zeros(N_atoms, dtype=torch.long, device=device)
        for i in range(N_atoms):
            sp = super_index[order[i]].item()
            local_rank[order[i]] = i - ptr[sp]
        
        # 初始化输出
        packed_point_idx = torch.full((M, n_sample), -1, dtype=torch.long, device=device)
        packed_mask = torch.zeros(M, n_sample, dtype=torch.bool, device=device)
        
        # 对于有点的超点，填充采样索引
        for sp_idx in range(M):
            n_pts = counts[sp_idx].item()
            if n_pts == 0:
                continue
            
            # 获取该超点的点
            start = ptr[sp_idx].item()
            end = ptr[sp_idx + 1].item()
            sp_points = order[start:end]
            
            # 随机选择（用 randperm 替代排序）
            if n_pts >= n_sample:
                perm = torch.randperm(n_pts, device=device)[:n_sample]
                sampled = sp_points[perm]
            else:
                # 重复填充
                repeat_times = (n_sample + n_pts - 1) // n_pts
                sampled = sp_points.repeat(repeat_times)[:n_sample]
            
            packed_point_idx[sp_idx] = sampled
            packed_mask[sp_idx] = True
    
    # 构建 packed_raw_points
    raw_tensors = []
    for key in raw_keys:
        if hasattr(nag[0], key):
            attr = getattr(nag[0], key)
            if attr is not None:
                if attr.dim() == 1:
                    attr = attr.unsqueeze(-1)
                raw_tensors.append(attr)
    
    if not raw_tensors:
        raise ValueError(f"无法获取原始点属性: {raw_keys}")
    
    # 拼接所有属性
    raw_data = torch.cat(raw_tensors, dim=-1) if len(raw_tensors) > 1 else raw_tensors[0]
    d_raw = raw_data.shape[-1]
    
    # Gather 构建 packed_raw_points
    safe_idx = packed_point_idx.clamp(min=0)
    packed_raw_points = raw_data[safe_idx.view(-1)].view(M, n_sample, d_raw)
    
    # 对 padding 位置置零
    packed_raw_points[~packed_mask] = 0.0
    
    return packed_raw_points, packed_point_idx, packed_mask


def run_hspt(
    aus: AdaptiveUncertaintySampler,
    cafm: CrossAttentionFusionModule,
    rrh: ResidualRefinementHead,
    sp_features: torch.Tensor,
    sp_centroids: torch.Tensor,
    coarse_logits: torch.Tensor,
    packed_raw_points: torch.Tensor,
    packed_mask: Optional[torch.Tensor] = None,
    handcrafted_features: Optional[torch.Tensor] = None,
    use_residual: bool = True
) -> HSPTOutput:
    """
    运行 H-SPT 三模块串联流水线。
    
    参数:
        aus: AdaptiveUncertaintySampler 实例
        cafm: CrossAttentionFusionModule 实例
        rrh: ResidualRefinementHead 实例
        sp_features: [M, D] 超点特征
        sp_centroids: [M, 3] 超点质心
        coarse_logits: [M, C] 粗分类 logits
        packed_raw_points: [M, N, d_raw] 打包的原始点张量
        packed_mask: [M, N] 有效点掩码（可选）
        handcrafted_features: [M, D_geo] 几何特征（可选）
        use_residual: 是否使用残差模式
    
    返回:
        HSPTOutput: 包含融合特征、困难索引、点级 logits 等
    """
    device = sp_features.device
    M = sp_features.shape[0]
    
    # ===========================
    # Module 1: AUS - 困难超点筛选
    # ===========================
    hard_sp_indices, scores = aus(coarse_logits, handcrafted_features)
    K = hard_sp_indices.numel()
    
    # 处理 K=0 边界
    if K == 0:
        N = packed_raw_points.shape[1] if packed_raw_points.dim() == 3 else 0
        C = coarse_logits.shape[1]
        D = sp_features.shape[1]
        return HSPTOutput(
            fused_features=sp_features,  # 无增强，返回原始
            hard_sp_indices=hard_sp_indices,
            point_logits=torch.zeros(0, N, C, device=device),
            packed_point_idx=None,
            packed_mask=None
        )
    
    # ===========================
    # Module 2: CAFM - 特征融合
    # ===========================
    enhanced_k, point_feats_k, fused_global = cafm(
        hard_sp_indices,
        sp_features,
        sp_centroids,
        packed_raw_points,
        key_padding_mask=~packed_mask[hard_sp_indices] if packed_mask is not None else None
    )
    
    # ===========================
    # Module 3: RRH - 点级细化
    # ===========================
    if use_residual:
        coarse_logits_k = coarse_logits[hard_sp_indices]
    else:
        coarse_logits_k = None
    
    point_logits = rrh(enhanced_k, point_feats_k, coarse_logits_k)
    
    return HSPTOutput(
        fused_features=fused_global,
        hard_sp_indices=hard_sp_indices,
        point_logits=point_logits,
        packed_point_idx=None,  # 由调用方维护
        packed_mask=packed_mask[hard_sp_indices] if packed_mask is not None else None
    )


class HSPTModule(nn.Module):
    """
    H-SPT 封装模块，用于集成到 SemanticSegmentationModule。
    
    将三个子模块封装为单个 nn.Module，便于参数管理和序列化。
    
    参数:
        d_model: 特征维度，默认 64
        num_classes: 分类类别数，默认 13
        topk_ratio: AUS 采样比例，默认 0.2
        n_sample: 每个超点采样的点数，默认 64
        d_raw: 原始点特征维度，默认 6
        n_heads: CAFM 注意力头数，默认 4
        use_residual: 是否使用残差模式，默认 True
        dropout: Dropout 比率，默认 0.1
    """
    
    def __init__(
        self,
        d_model: int = 64,
        num_classes: int = 13,
        topk_ratio: float = 0.2,
        n_sample: int = 64,
        d_raw: int = 6,
        n_heads: int = 4,
        use_residual: bool = True,
        dropout: float = 0.1,
        alpha: float = 0.7,
        beta: float = 0.3
    ):
        super().__init__()
        
        self.d_model = d_model
        self.num_classes = num_classes
        self.n_sample = n_sample
        self.use_residual = use_residual
        
        # 三个子模块
        self.aus = AdaptiveUncertaintySampler(
            topk_ratio=topk_ratio,
            alpha=alpha,
            beta=beta
        )
        
        self.cafm = CrossAttentionFusionModule(
            d_model=d_model,
            n_heads=n_heads,
            d_raw=d_raw,
            dropout=dropout
        )
        
        self.rrh = ResidualRefinementHead(
            d_model=d_model,
            num_classes=num_classes,
            dropout=dropout
        )
    
    def forward(
        self,
        sp_features: torch.Tensor,
        sp_centroids: torch.Tensor,
        coarse_logits: torch.Tensor,
        packed_raw_points: torch.Tensor,
        packed_mask: Optional[torch.Tensor] = None,
        handcrafted_features: Optional[torch.Tensor] = None
    ) -> HSPTOutput:
        """
        前向传播。
        
        参数:
            sp_features: [M, D] 超点特征
            sp_centroids: [M, 3] 超点质心
            coarse_logits: [M, C] 粗分类 logits
            packed_raw_points: [M, N, d_raw] 打包的原始点张量
            packed_mask: [M, N] 有效点掩码
            handcrafted_features: [M, D_geo] 几何特征
        
        返回:
            HSPTOutput
        """
        return run_hspt(
            self.aus,
            self.cafm,
            self.rrh,
            sp_features,
            sp_centroids,
            coarse_logits,
            packed_raw_points,
            packed_mask,
            handcrafted_features,
            self.use_residual
        )


def compute_refine_loss(
    point_logits: torch.Tensor,
    packed_point_idx: torch.Tensor,
    hard_sp_indices: torch.Tensor,
    gt_labels: torch.Tensor,
    packed_mask: Optional[torch.Tensor] = None,
    num_classes: int = 13,
    ignore_index: int = -100
) -> torch.Tensor:
    """
    计算点级细化损失。
    
    参数:
        point_logits: [K, N, C] 点级预测 logits
        packed_point_idx: [M, N] 或 [K, N] 采样点索引
        hard_sp_indices: [K] 困难超点索引
        gt_labels: [N_total] 原始点云 GT 标签
        packed_mask: [K, N] 有效点掩码
        num_classes: 类别数
        ignore_index: 忽略索引
    
    返回:
        loss: 标量 refine loss
    """
    K, N, C = point_logits.shape
    device = point_logits.device
    
    if K == 0:
        return torch.tensor(0.0, device=device, requires_grad=True)
    
    # 获取对应的采样点索引
    if packed_point_idx.shape[0] != K:
        # 需要用 hard_sp_indices 索引
        packed_idx_k = packed_point_idx[hard_sp_indices]
    else:
        packed_idx_k = packed_point_idx
    
    # 获取 GT 标签
    safe_idx = packed_idx_k.clamp(min=0)
    gt_point_labels = gt_labels[safe_idx.view(-1)].view(K, N)
    
    # 对 padding 位置设为 ignore_index
    if packed_mask is not None:
        gt_point_labels[~packed_mask] = ignore_index
    
    # 对 -1 索引位置设为 ignore_index
    gt_point_labels[packed_idx_k < 0] = ignore_index
    
    # 展平计算 CE
    logits_flat = point_logits.view(-1, C)
    labels_flat = gt_point_labels.view(-1)
    
    loss = F.cross_entropy(logits_flat, labels_flat, ignore_index=ignore_index)
    
    return loss
