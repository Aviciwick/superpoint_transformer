"""
残差细化头模块 (Residual Refinement Head, RRH)

模块功能：
    作为 H-SPT 系统的"尾部"，打破超点的物理边界限制，
    对困难超点内部的原始点进行点级分类，实现"亚超点级"精度。

设计范式：
    Point-Level Residual Learning (Shared MLP)

架构位置：
    位于 Module 2 (CAFM) 之后，是 H-SPT 核心链路的最后一环。
    - 输入：CAFM 增强后的超点特征 + 编码后的点特征 + 原始 Logits
    - 输出：点级分类 Logits

核心算法：
    1. 特征广播 (Broadcasting): 将超点特征复制 N 份对齐到点级
    2. 特征拼接 (Concatenation): 结合宏观上下文与微观细节
    3. 残差预测 (Residual Prediction): MLP 预测点级 Logits 偏移量
    4. 残差加和 (Residual Addition): Final = Base + Delta
"""

import warnings
import torch
import torch.nn as nn
from typing import Optional


__all__ = ['ResidualRefinementHead']


class ResidualRefinementHead(nn.Module):
    """
    残差细化头 - 点级分类器。
    
    该模块接收 Module 2 增强后的超点特征和点特征，
    通过 Shared MLP 预测每个原始点的语义 Logits 偏移量，
    从而修正超点内部的边界分割错误。
    
    核心设计：
        1. 强制残差学习：Final = Coarse + Delta（网络只学习偏移量）
        2. 特征融合：Concat(超点特征, 点特征) 提供完整的上下文信息
        3. 轻量级 MLP：保持计算效率
    
    参数:
        d_model (int): 特征维度（需与 Module 2 一致），默认 64
        num_classes (int): 分类类别数，默认 13 (S3DIS)
        hidden_dim (int): MLP 中间层维度，默认等于 d_model
        dropout (float): Dropout 比率，默认 0.1
    
    示例:
        >>> head = ResidualRefinementHead(d_model=64, num_classes=13)
        >>> enhanced = torch.randn(20, 64)      # K=20 个超点
        >>> points = torch.randn(20, 32, 64)    # 每个超点 N=32 个点
        >>> coarse = torch.randn(20, 13)        # 原始超点 Logits
        >>> point_logits = head(enhanced, points, coarse)
        >>> print(point_logits.shape)  # [20, 32, 13]
    """
    
    def __init__(
        self,
        d_model: int = 64,
        num_classes: int = 13,
        hidden_dim: Optional[int] = None,
        dropout: float = 0.1
    ):
        """
        初始化残差细化头。
        
        参数:
            d_model: 特征维度，默认 64
            num_classes: 分类类别数，默认 13
            hidden_dim: MLP 中间层维度，默认等于 d_model
            dropout: Dropout 比率，默认 0.1
        """
        super().__init__()
        
        # 参数验证
        assert d_model > 0, f"d_model 必须为正整数，当前值: {d_model}"
        assert num_classes > 0, f"num_classes 必须为正整数，当前值: {num_classes}"
        
        self.d_model = d_model
        self.num_classes = num_classes
        hidden_dim = hidden_dim or d_model
        
        # ===========================
        # 残差预测 MLP
        # ===========================
        # 输入: [K, N, 2*D] (超点特征 + 点特征拼接)
        # 输出: [K, N, C] (delta logits)
        self.refinement_head = nn.Sequential(
            nn.Linear(d_model * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),  # inplace 优化显存
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )
    
    def forward(
        self,
        enhanced_sp_features: torch.Tensor,
        point_features_K: torch.Tensor,
        coarse_logits: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        前向传播：预测点级分类 Logits。
        
        参数:
            enhanced_sp_features: 形状 [K, D]，来自 Module 2 的增强超点特征
            point_features_K: 形状 [K, N, D]，来自 Module 2 的编码点特征
            coarse_logits: 形状 [K, C]，SPT Decoder 的原始超点预测（可选）
                         - 如果提供：采用残差策略 Final = Base + Delta
                         - 如果不提供：采用纯预测模式 Final = Delta
        
        返回:
            point_logits: 形状 [K, N, C]，每个原始点的最终分类 Logits
        
        说明:
            - Loss 计算不在本模块内，需在外部使用 GT 点标签计算
        """
        device = enhanced_sp_features.device
        dtype = enhanced_sp_features.dtype
        K = enhanced_sp_features.shape[0]
        
        # ===========================
        # 步骤 0: 边界与输入验证
        # ===========================
        # K==0 边界处理：直接返回空张量
        if K == 0:
            N = point_features_K.shape[1] if point_features_K.dim() == 3 else 0
            return torch.zeros(0, N, self.num_classes, device=device, dtype=dtype)
        
        # 获取维度信息
        N = point_features_K.shape[1]
        D = point_features_K.shape[2]
        
        # 形状验证
        if enhanced_sp_features.shape[1] != D:
            raise ValueError(
                f"enhanced_sp_features 维度 ({enhanced_sp_features.shape[1]}) "
                f"与 point_features_K 维度 ({D}) 不匹配"
            )
        
        # coarse_logits 验证（若提供）
        use_residual = coarse_logits is not None
        if use_residual:
            if coarse_logits.shape[0] != K:
                raise ValueError(
                    f"coarse_logits 形状不匹配：期望 K={K}，实际 {coarse_logits.shape[0]}"
                )
            if coarse_logits.shape[1] != self.num_classes:
                raise ValueError(
                    f"coarse_logits 类别数 ({coarse_logits.shape[1]}) "
                    f"与 num_classes ({self.num_classes}) 不匹配"
                )
            # 设备对齐
            if coarse_logits.device != device:
                warnings.warn(
                    f"coarse_logits 设备 ({coarse_logits.device}) "
                    f"与 enhanced_sp_features 设备 ({device}) 不一致，已自动对齐。",
                    RuntimeWarning
                )
                coarse_logits = coarse_logits.to(device)
        
        # point_features_K 设备对齐
        if point_features_K.device != device:
            warnings.warn(
                f"point_features_K 设备 ({point_features_K.device}) "
                f"与 enhanced_sp_features 设备 ({device}) 不一致，已自动对齐。",
                RuntimeWarning
            )
            point_features_K = point_features_K.to(device)
        
        # ===========================
        # 步骤 1: 特征广播 (Broadcasting)
        # ===========================
        # [K, D] -> [K, 1, D] -> [K, N, D]
        sp_expanded = enhanced_sp_features.unsqueeze(1).expand(-1, N, -1)
        
        # ===========================
        # 步骤 2: 特征拼接 (Concatenation)
        # ===========================
        # Cat([K, N, D], [K, N, D]) -> [K, N, 2D]
        combined_features = torch.cat([sp_expanded, point_features_K], dim=-1)
        
        # ===========================
        # 步骤 3: 预测残差 Logits
        # ===========================
        # [K, N, 2D] -> [K, N, C]
        delta_logits = self.refinement_head(combined_features)
        
        # ===========================
        # 步骤 4: 残差加和（可选）
        # ===========================
        if use_residual:
            # Base Logits: [K, C] -> [K, 1, C] (广播到 N 个点)
            base_logits = coarse_logits.unsqueeze(1)
            # Final = Base + Delta
            point_logits = base_logits + delta_logits
        else:
            # 纯预测模式
            point_logits = delta_logits
        
        return point_logits
    
    def extra_repr(self) -> str:
        """返回模块的额外表示信息。"""
        return f"d_model={self.d_model}, num_classes={self.num_classes}"


def demo_usage():
    """演示用法：展示如何使用 ResidualRefinementHead。"""
    print("=" * 60)
    print("ResidualRefinementHead 演示")
    print("=" * 60)
    
    # 创建模块
    head = ResidualRefinementHead(
        d_model=64,
        num_classes=13,
        dropout=0.1
    )
    print(f"\n模块配置: {head}")
    
    # 模拟输入数据
    K = 20    # 困难超点数
    N = 32    # 每个超点的采样点数
    D = 64    # 特征维度
    C = 13    # 类别数
    
    torch.manual_seed(42)
    enhanced_sp = torch.randn(K, D)
    point_feats = torch.randn(K, N, D)
    coarse = torch.randn(K, C)
    
    print(f"\n输入形状:")
    print(f"  enhanced_sp_features: {enhanced_sp.shape}")
    print(f"  point_features_K: {point_feats.shape}")
    print(f"  coarse_logits: {coarse.shape}")
    
    # 前向传播
    point_logits = head(enhanced_sp, point_feats, coarse)
    
    print(f"\n输出形状:")
    print(f"  point_logits: {point_logits.shape}")
    
    # 验证残差加和
    print("\n残差策略验证:")
    print(f"  点级预测已结合超点 Base Logits")
    
    print("\n" + "=" * 60)
    print("演示完成!")
    print("=" * 60)


if __name__ == "__main__":
    demo_usage()
