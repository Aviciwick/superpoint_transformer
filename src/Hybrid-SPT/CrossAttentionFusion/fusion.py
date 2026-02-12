"""
点-超点交叉注意力融合模块 (Cross-Attention Fusion Module, CAFM)

模块功能：
    针对困难超点，通过 Cross-Attention 机制让超点特征主动聚合其内部
    原始点云的几何和颜色信息，从而恢复被池化操作丢失的细粒度特征。

设计范式：
    In-Loop Feature Enhancement (可插拔增强模块)

架构位置：
    位于 Module 1 (AUS) 之后，Module 3 (Refinement Head) 之前。
    - 输入：AUS 筛选的困难超点索引 + SPT Decoder 特征
    - 输出：增强后的超点特征 + 点级特征（供 Module 3 复用）

核心算法：
    1. 坐标局部化 (Canonicalization): 将绝对坐标转换为相对于超点质心的局部坐标
    2. 点云编码 (Mini-PointNet): 将原始点数据映射到高维特征空间
    3. 交叉注意力 (Cross-Attention): Query=超点, Key/Value=原始点
    4. 残差融合 (Residual Fusion): 增强特征加回原特征
    5. 全局回填 (Scatter Update): 将增强特征写回全局特征图
"""

import math
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


__all__ = ['CrossAttentionFusionModule']


class CrossAttentionFusionModule(nn.Module):
    """
    点-超点交叉注意力融合模块。
    
    该模块接收 Module 1 筛选出的困难超点，提取其内部原始点云，
    通过 Cross-Attention 进行特征增强，并更新全局特征图。
    
    关键特性：
        1. Canonicalization: 使用局部相对坐标，消除绝对位置影响
        2. Tensor-based Input: 假设输入点云已在 Dataloader 阶段打包为 [M, N, D_raw]
        3. Global Update: 支持将增强特征回填至全图，保持梯度流
        4. Point Feature Reuse: 输出点级特征供 Module 3 直接使用
    
    参数:
        d_model (int): 超点特征维度（需与 SPT Backbone 一致），默认 64
        d_raw (int): 原始点云维度（通常为 6: XYZ(3) + RGB(3)），默认 6
        n_heads (int): 注意力头数，默认 4
        dropout (float): Dropout 比率，默认 0.1
        use_ffn (bool): 是否使用 Feed-Forward Network，默认 False
    
    示例:
        >>> module = CrossAttentionFusionModule(d_model=64, n_heads=4)
        >>> hard_indices = torch.tensor([0, 5, 10])
        >>> all_features = torch.randn(100, 64)
        >>> centroids = torch.randn(100, 3)
        >>> packed_points = torch.randn(100, 32, 6)
        >>> enhanced, point_feats, fused = module(hard_indices, all_features, centroids, packed_points)
    """
    
    def __init__(
        self,
        d_model: int = 64,
        d_raw: int = 6,
        n_heads: int = 4,
        dropout: float = 0.1,
        use_ffn: bool = False
    ):
        """
        初始化交叉注意力融合模块。
        
        参数:
            d_model: 超点特征维度，默认 64
            d_raw: 原始点云维度（XYZ+RGB），默认 6
            n_heads: 注意力头数，默认 4
            dropout: Dropout 比率，默认 0.1
            use_ffn: 是否使用 FFN 进一步增强，默认 False
        """
        super().__init__()
        
        # 参数验证
        assert d_model > 0, f"d_model 必须为正整数，当前值: {d_model}"
        assert d_model % n_heads == 0, \
            f"d_model ({d_model}) 必须能被 n_heads ({n_heads}) 整除"
        
        self.d_model = d_model
        self.d_raw = d_raw
        self.n_heads = n_heads
        
        # ===========================
        # 1. 点云编码器 (Mini-PointNet)
        # ===========================
        # 将 (Local_XYZ + RGB) 映射到 d_model 维度
        self.point_encoder = nn.Sequential(
            nn.Linear(d_raw, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(inplace=True),
            nn.Linear(d_model, d_model)
        )
        
        # ===========================
        # 2. 交叉注意力组件
        # ===========================
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True  # 使用 [Batch, Seq, Dim] 格式
        )
        
        # ===========================
        # 3. 残差融合层
        # ===========================
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
        # ===========================
        # 4. 可选 FFN 层
        # ===========================
        self.use_ffn = use_ffn
        if use_ffn:
            self.ffn = nn.Sequential(
                nn.Linear(d_model, d_model * 4),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(d_model * 4, d_model)
            )
            self.norm2 = nn.LayerNorm(d_model)
    
    def forward(
        self,
        hard_sp_indices: torch.Tensor,
        all_sp_features: torch.Tensor,
        all_sp_centroids: torch.Tensor,
        packed_raw_points: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        前向传播：对困难超点进行特征增强。
        
        参数:
            hard_sp_indices: 形状 [K]，来自 Module 1 的困难超点索引
            all_sp_features: 形状 [M, D]，SPT Decoder 输出的全局超点特征
            all_sp_centroids: 形状 [M, 3]，所有超点的物理质心
            packed_raw_points: 形状 [M, N, D_raw]，预打包的原始点云（XYZ在前3维）
        
        返回:
            enhanced_features_K: 形状 [K, D]，仅增强的 K 个超点特征
            point_features_K: 形状 [K, N, D]，编码后的点特征（供 Module 3 复用）
            fused_global_features: 形状 [M, D]，全局更新版特征图
        
        说明:
            - 若 hard_sp_indices 为空，直接返回原始特征
            - fused_global_features 中未选中区域保持原样，选中区域已更新
        """
        M = all_sp_features.shape[0]
        device = all_sp_features.device
        dtype = all_sp_features.dtype
        K = hard_sp_indices.numel()
        N = packed_raw_points.shape[1] if packed_raw_points.dim() == 3 else 0
        
        # ===========================
        # 步骤 0: 边界与输入验证
        # ===========================
        # M==0 边界处理
        if M == 0:
            empty_enhanced = torch.zeros(0, self.d_model, device=device, dtype=dtype)
            empty_points = torch.zeros(0, 0, self.d_model, device=device, dtype=dtype)
            return empty_enhanced, empty_points, all_sp_features
        
        # K==0 边界处理：没有采样到任何点，直接返回原始特征
        if K == 0:
            empty_enhanced = torch.zeros(0, self.d_model, device=device, dtype=dtype)
            empty_points = torch.zeros(0, N, self.d_model, device=device, dtype=dtype)
            return empty_enhanced, empty_points, all_sp_features
        
        # 输入形状验证
        if packed_raw_points.dim() != 3:
            raise ValueError(
                f"packed_raw_points 必须是 3D 张量 [M, N, D_raw]，"
                f"实际维度: {packed_raw_points.dim()}"
            )
        if packed_raw_points.shape[-1] < 3:
            raise ValueError(
                f"packed_raw_points 最后维度必须 >= 3（至少包含 XYZ），"
                f"实际: {packed_raw_points.shape[-1]}"
            )
        # d_raw 不匹配时直接报错（Linear 层会崩溃，提前检测）
        if packed_raw_points.shape[-1] != self.d_raw:
            raise ValueError(
                f"packed_raw_points 维度 ({packed_raw_points.shape[-1]}) "
                f"与 d_raw ({self.d_raw}) 不匹配，请检查 Dataloader 或模块初始化参数"
            )
        
        # ===========================
        # 索引校验（dtype + 越界）
        # ===========================
        # 强制 dtype 为 torch.long
        if hard_sp_indices.dtype != torch.long:
            hard_sp_indices = hard_sp_indices.to(torch.long)
        
        # 越界校验
        if K > 0:
            idx_min, idx_max = hard_sp_indices.min().item(), hard_sp_indices.max().item()
            if idx_min < 0 or idx_max >= M:
                raise IndexError(
                    f"hard_sp_indices 越界：索引范围 [{idx_min}, {idx_max}]，"
                    f"有效范围 [0, {M-1}]"
                )
        
        # hard_sp_indices 设备对齐
        if hard_sp_indices.device != device:
            warnings.warn(
                f"hard_sp_indices 设备 ({hard_sp_indices.device}) "
                f"与 all_sp_features 设备 ({device}) 不一致，已自动对齐。",
                RuntimeWarning
            )
            hard_sp_indices = hard_sp_indices.to(device)
        
        # 设备对齐检查
        if all_sp_centroids.device != device:
            warnings.warn(
                f"all_sp_centroids 设备 ({all_sp_centroids.device}) "
                f"与 all_sp_features 设备 ({device}) 不一致，已自动对齐。",
                RuntimeWarning
            )
            all_sp_centroids = all_sp_centroids.to(device)
        
        if packed_raw_points.device != device:
            warnings.warn(
                f"packed_raw_points 设备 ({packed_raw_points.device}) "
                f"与 all_sp_features 设备 ({device}) 不一致，已自动对齐。",
                RuntimeWarning
            )
            packed_raw_points = packed_raw_points.to(device)
        
        # ===========================
        # 步骤 1: 数据提取 (Slicing)
        # ===========================
        sp_feat_k = all_sp_features[hard_sp_indices]      # [K, D]
        centroids_k = all_sp_centroids[hard_sp_indices]   # [K, 3]
        raw_points_k = packed_raw_points[hard_sp_indices] # [K, N, D_raw]
        
        # ===========================
        # 步骤 2: 坐标局部化 (Canonicalization)
        # ===========================
        # 分离 XYZ 和 颜色/属性
        raw_xyz = raw_points_k[..., :3]   # [K, N, 3]
        raw_attr = raw_points_k[..., 3:]  # [K, N, D_raw-3]
        
        # 关键步骤：减去质心 -> 局部坐标
        # [K, N, 3] - [K, 1, 3] -> [K, N, 3]
        local_xyz = raw_xyz - centroids_k.unsqueeze(1)
        
        # 重新拼接：局部坐标 + 属性
        point_input = torch.cat([local_xyz, raw_attr], dim=-1)  # [K, N, D_raw]
        
        # ===========================
        # 步骤 3: 点云编码 (Mini-PointNet)
        # ===========================
        # Encoding: [K, N, D_raw] -> [K, N, D]
        # 保存这个 point_features_k 供 Module 3 使用！
        point_features_k = self.point_encoder(point_input)
        
        # ===========================
        # 步骤 4: 交叉注意力 (Cross-Attention)
        # ===========================
        # Query: 超点特征 [K, D] -> [K, 1, D]
        # Key/Value: 点特征 [K, N, D]
        query = sp_feat_k.unsqueeze(1)
        
        attn_out, _ = self.cross_attn(
            query=query,
            key=point_features_k,
            value=point_features_k
        )
        # attn_out: [K, 1, D]
        
        # ===========================
        # 步骤 5: 残差融合
        # ===========================
        # Squeeze dim 1: [K, 1, D] -> [K, D]
        attn_out = attn_out.squeeze(1)
        
        # 残差连接 + LayerNorm
        enhanced_feat_k = self.norm1(sp_feat_k + self.dropout(attn_out))
        
        # 可选 FFN
        if self.use_ffn:
            ffn_out = self.ffn(enhanced_feat_k)
            enhanced_feat_k = self.norm2(enhanced_feat_k + ffn_out)
        
        # ===========================
        # 步骤 6: 全局回填 (Scatter Update)
        # ===========================
        # 复制一份全局特征用于更新（保持梯度流）
        fused_global_features = all_sp_features.clone()
        
        # 将增强后的特征填回对应的索引位置
        # PyTorch 的索引赋值操作支持梯度回传
        fused_global_features[hard_sp_indices] = enhanced_feat_k
        
        return enhanced_feat_k, point_features_k, fused_global_features
    
    def extra_repr(self) -> str:
        """返回模块的额外表示信息。"""
        return (
            f"d_model={self.d_model}, d_raw={self.d_raw}, "
            f"n_heads={self.n_heads}, use_ffn={self.use_ffn}"
        )


def demo_usage():
    """演示用法：展示如何使用 CrossAttentionFusionModule。"""
    print("=" * 60)
    print("CrossAttentionFusionModule 演示")
    print("=" * 60)
    
    # 创建模块
    module = CrossAttentionFusionModule(
        d_model=64,
        d_raw=6,
        n_heads=4,
        dropout=0.1,
        use_ffn=False
    )
    print(f"\n模块配置: {module}")
    
    # 模拟输入数据
    M = 100   # 超点总数
    K = 20    # 困难超点数
    N = 32    # 每个超点的采样点数
    D = 64    # 特征维度
    
    torch.manual_seed(42)
    hard_sp_indices = torch.randperm(M)[:K]
    all_sp_features = torch.randn(M, D)
    all_sp_centroids = torch.randn(M, 3)
    packed_raw_points = torch.randn(M, N, 6)
    
    print(f"\n输入形状:")
    print(f"  hard_sp_indices: {hard_sp_indices.shape}")
    print(f"  all_sp_features: {all_sp_features.shape}")
    print(f"  all_sp_centroids: {all_sp_centroids.shape}")
    print(f"  packed_raw_points: {packed_raw_points.shape}")
    
    # 前向传播
    enhanced, point_feats, fused = module(
        hard_sp_indices, all_sp_features, all_sp_centroids, packed_raw_points
    )
    
    print(f"\n输出形状:")
    print(f"  enhanced_features_K: {enhanced.shape}")
    print(f"  point_features_K: {point_feats.shape}")
    print(f"  fused_global_features: {fused.shape}")
    
    # 验证全局回填
    diff_count = (fused != all_sp_features).any(dim=1).sum().item()
    print(f"\n全局回填验证:")
    print(f"  被更新的超点数: {diff_count} (应等于 K={K})")
    
    print("\n" + "=" * 60)
    print("演示完成!")
    print("=" * 60)


if __name__ == "__main__":
    demo_usage()
