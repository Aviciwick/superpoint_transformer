"""
自适应不确定性采样器 (Adaptive Uncertainty Sampler, AUS)

模块功能：
    基于规则的困难超点筛选模块。结合语义不确定性（香农熵）和几何复杂性（散射度），
    筛选出 Top-K% 最难的超点，用于后续精细化处理。

设计范式：
    Heuristic / Rule-Based (Non-learnable)
    
架构位置：
    位于 SPT Decoder 末端，作为后续 Cross-Attention 和 Refinement Head 模块的"守门员"。

数据流依赖说明：
    若后续模块需要获取困难超点内部的原始点（Raw Points），
    需在 Dataset 的 __getitem__ 中预先构建 `sp_to_raw_indices` 映射表。
    本模块仅输出超点索引，不涉及此映射的实现。
    
几何特征索引参考（SPT 默认顺序）：
    [0] linearity  - 线性度
    [1] planarity  - 平面度
    [2] scattering - 散射度 (默认使用)
    [3] verticality - 垂直度
    ...
"""

import math
import warnings
import torch
import torch.nn as nn
from typing import Tuple, Optional


__all__ = ['AdaptiveUncertaintySampler']


class AdaptiveUncertaintySampler(nn.Module):
    """
    自适应不确定性采样器 - 基于规则的困难超点筛选模块。
    
    该模块充当"过滤器"，计算每个超点的"难度得分"，并筛选出 Top-K% 最难的超点。
    算法结合语义不确定性（熵）和几何复杂性（散射度）进行加权打分，无需额外的可学习参数。
    
    核心算法：
        1. 语义评估：基于分类器输出的 Logits 计算香农熵，衡量模型对该超点语义识别的困惑度
        2. 几何评估：从预计算的手工特征中提取散射度，衡量该超点内部的物理几何复杂性
        3. 加权融合：S = alpha * H_norm + beta * G_norm
        4. Top-K 采样：选取得分最高的 K 个超点
    
    参数:
        topk_ratio (float): 采样比例，默认 0.20 表示选取前 20%
        alpha (float): 语义熵权重，默认 0.7
        beta (float): 几何特征权重，默认 0.3
        scatter_idx (int): 几何特征张量中散射度的通道索引，默认 2
        eps (float): 数值稳定性参数，防止除零错误，默认 1e-6
    
    示例:
        >>> sampler = AdaptiveUncertaintySampler(topk_ratio=0.2, alpha=0.7, beta=0.3)
        >>> logits = torch.randn(100, 20)  # 100 个超点，20 个类别
        >>> features = torch.rand(100, 9)   # 100 个超点，9 维几何特征
        >>> hard_indices, scores = sampler(logits, features)
        >>> print(f"采样了 {len(hard_indices)} 个困难超点")
    """
    
    def __init__(
        self,
        topk_ratio: float = 0.20,
        alpha: float = 0.7,
        beta: float = 0.3,
        scatter_idx: int = 2,
        eps: float = 1e-6
    ):
        """
        初始化自适应不确定性采样器。
        
        参数:
            topk_ratio: 采样比例，取值范围 (0, 1]，默认 0.20
            alpha: 语义熵权重，默认 0.7
            beta: 几何特征权重，默认 0.3
            scatter_idx: 散射度在 handcrafted_features 中的通道索引，默认 2
            eps: 数值稳定性参数，默认 1e-6
        """
        super().__init__()
        
        # 参数验证
        assert 0.0 < topk_ratio <= 1.0, \
            f"topk_ratio 必须在 (0, 1] 范围内，当前值: {topk_ratio}"
        assert alpha >= 0 and beta >= 0, \
            f"alpha 和 beta 必须非负，当前值: alpha={alpha}, beta={beta}"
        
        # 存储超参数（不注册为参数，因为无需梯度）
        self.topk_ratio = topk_ratio
        self.alpha = alpha
        self.beta = beta
        self.scatter_idx = scatter_idx
        self.eps = eps
    
    def forward(
        self,
        coarse_logits: torch.Tensor,
        handcrafted_features: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播：计算难度得分并筛选困难超点。
        
        参数:
            coarse_logits: 形状 [M, C]，SPT Decoder Level-1 的原始输出（未过 Softmax）
                          M 为当前 Batch 超点总数，C 为类别数
            handcrafted_features: 形状 [M, D]，预计算的几何特征（可选）
                                 若为 None，则自动回退到仅使用语义熵
        
        返回:
            hard_sp_indices: 形状 [K]，被选中的 Top-K 个超点索引（torch.int64）
            debug_scores: 形状 [M]，所有超点的评分，用于可视化调试
        
        说明:
            - 本方法不涉及参数学习，所有操作均为确定性计算
            - 采样逻辑不需要梯度，但不会阻断 coarse_logits 的梯度回传
        """
        M = coarse_logits.shape[0]  # 超点总数
        C = coarse_logits.shape[1]  # 类别数
        device = coarse_logits.device
        dtype = coarse_logits.dtype
        
        # ===============================
        # 步骤 0: 边界与输入验证
        # ===============================
        # coarse_logits 维度校验
        if coarse_logits.dim() != 2:
            raise ValueError(
                f"coarse_logits 必须是 2D 张量 [M, C]，"
                f"实际维度: {coarse_logits.dim()}"
            )
        if C == 0:
            raise ValueError("coarse_logits 类别数 C 必须 > 0")
        
        # M==0 边界处理：直接返回空张量
        if M == 0:
            empty_indices = torch.zeros(0, dtype=torch.long, device=device)
            empty_scores = torch.zeros(0, dtype=dtype, device=device)
            return empty_indices, empty_scores
        
        # 验证 handcrafted_features 形状（若提供）
        if handcrafted_features is not None:
            if handcrafted_features.shape[0] != M:
                raise ValueError(
                    f"handcrafted_features 形状不匹配：期望 shape[0]={M}，"
                    f"实际 shape[0]={handcrafted_features.shape[0]}"
                )
        
        # 清洗 logits 中的 NaN/Inf（无条件执行，torch.compile 友好）
        coarse_logits = torch.nan_to_num(coarse_logits, nan=0.0, posinf=0.0, neginf=0.0)
        
        # ===============================
        # 步骤 1: 计算语义不确定性（香农熵）
        # ===============================
        # Softmax 得到概率分布
        probs = torch.softmax(coarse_logits, dim=1)  # [M, C]
        
        # 计算香农熵: H = -Σ P·log(P)
        # 使用 eps 防止 log(0) 产生 NaN
        log_probs = torch.log(probs + self.eps)
        entropy = -torch.sum(probs * log_probs, dim=1)  # [M]
        
        # 归一化熵到 [0, 1] 范围
        # 最大熵为 log(C)（均匀分布时）
        max_entropy = math.log(C)
        entropy_norm = entropy / (max_entropy + self.eps)  # [M]
        
        # ===============================
        # 步骤 2: 获取几何显著性
        # ===============================
        if handcrafted_features is not None and self.beta > 0:
            # 设备对齐检查
            if handcrafted_features.device != device:
                warnings.warn(
                    f"handcrafted_features 设备 ({handcrafted_features.device}) "
                    f"与 coarse_logits 设备 ({device}) 不一致，已自动对齐。",
                    RuntimeWarning
                )
                handcrafted_features = handcrafted_features.to(device)
            
            # 提取散射度特征
            if self.scatter_idx < handcrafted_features.shape[1]:
                geo_raw = handcrafted_features[:, self.scatter_idx]  # [M]
            else:
                # 索引越界时回退并发出警告
                warnings.warn(
                    f"scatter_idx ({self.scatter_idx}) 超出 handcrafted_features "
                    f"维度 ({handcrafted_features.shape[1]})，几何特征将被忽略。",
                    RuntimeWarning
                )
                geo_raw = torch.zeros(M, device=device, dtype=dtype)
            
            # 处理 NaN/Inf 值：填充为 0
            geo_raw = torch.nan_to_num(geo_raw, nan=0.0, posinf=0.0, neginf=0.0)
            
            # Batch 内 Min-Max 归一化（关键操作，确保不同场景间的可比性）
            geo_min = geo_raw.min()
            geo_max = geo_raw.max()
            geo_norm = (geo_raw - geo_min) / (geo_max - geo_min + self.eps)  # [M]
            
            # 使用配置的 beta 作为有效权重
            effective_beta = self.beta
        else:
            # 回退机制：若几何特征不可用，仅使用语义熵
            geo_norm = torch.zeros(M, device=device, dtype=entropy_norm.dtype)
            effective_beta = 0.0
        
        # ===============================
        # 步骤 3: 加权融合评分
        # ===============================
        # 直接使用配置的 alpha（不做归一化，由用户控制权重比例）
        effective_alpha = self.alpha
        
        # 总得分 = alpha * 语义熵 + beta * 几何特征
        total_scores = effective_alpha * entropy_norm + effective_beta * geo_norm  # [M]
        
        # ===============================
        # 步骤 4: Top-K 采样
        # ===============================
        # 计算 K，使用 ceil 向上取整，确保至少为 1（边界保护）
        # 根据设计文档：K = ceil(M * topk_ratio)
        K = max(1, math.ceil(M * self.topk_ratio))
        
        # 确保 K 不超过 M（当 topk_ratio > 1.0 被错误传入时的防御）
        K = min(K, M)
        
        # 使用 torch.topk 获取最高分的 K 个索引
        _, hard_sp_indices = torch.topk(total_scores, k=K, largest=True, sorted=False)
        
        return hard_sp_indices, total_scores
    
    def extra_repr(self) -> str:
        """
        返回模块的额外表示信息，用于打印模块时显示配置。
        """
        return (
            f"topk_ratio={self.topk_ratio}, "
            f"alpha={self.alpha}, beta={self.beta}, "
            f"scatter_idx={self.scatter_idx}"
        )


def demo_usage():
    """
    演示用法：展示如何使用 AdaptiveUncertaintySampler。
    """
    print("=" * 60)
    print("AdaptiveUncertaintySampler 演示")
    print("=" * 60)
    
    # 创建采样器
    sampler = AdaptiveUncertaintySampler(
        topk_ratio=0.20,
        alpha=0.7,
        beta=0.3,
        scatter_idx=2
    )
    print(f"\n采样器配置: {sampler}")
    
    # 模拟输入数据
    M = 100   # 超点数量
    C = 20    # 类别数
    D = 9     # 几何特征维度
    
    torch.manual_seed(42)
    coarse_logits = torch.randn(M, C)
    handcrafted_features = torch.rand(M, D)
    
    print(f"\n输入形状:")
    print(f"  coarse_logits: {coarse_logits.shape}")
    print(f"  handcrafted_features: {handcrafted_features.shape}")
    
    # 运行采样
    hard_indices, scores = sampler(coarse_logits, handcrafted_features)
    
    print(f"\n输出:")
    print(f"  采样超点数: {len(hard_indices)} (比例: {len(hard_indices)/M:.1%})")
    print(f"  索引范围: [{hard_indices.min().item()}, {hard_indices.max().item()}]")
    print(f"  评分范围: [{scores.min().item():.4f}, {scores.max().item():.4f}]")
    
    # 测试回退机制（无几何特征）
    print("\n测试回退机制（无几何特征）:")
    hard_indices_no_geo, scores_no_geo = sampler(coarse_logits, None)
    print(f"  采样超点数: {len(hard_indices_no_geo)}")
    print(f"  评分范围: [{scores_no_geo.min().item():.4f}, {scores_no_geo.max().item():.4f}]")
    
    print("\n" + "=" * 60)
    print("演示完成!")
    print("=" * 60)


if __name__ == "__main__":
    demo_usage()
