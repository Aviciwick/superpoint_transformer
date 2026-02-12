"""
H-SPT (Hybrid SuperPoint Transformer) 模块包

包含三个核心模块：
- AUS (AdaptiveUncertaintySampler): 自适应不确定性采样器
- CAFM (CrossAttentionFusionModule): 交叉注意力融合模块
- RRH (ResidualRefinementHead): 残差细化头

以及用于训练/推理的 pipeline 工具函数。
"""

from .aus import AdaptiveUncertaintySampler
from .cafm import CrossAttentionFusionModule
from .rrh import ResidualRefinementHead
from .pipeline import (
    build_packed_points,
    run_hspt,
    HSPTOutput,
    HSPTModule,
    compute_refine_loss
)

__all__ = [
    'AdaptiveUncertaintySampler',
    'CrossAttentionFusionModule',
    'ResidualRefinementHead',
    'build_packed_points',
    'run_hspt',
    'HSPTOutput',
    'HSPTModule',
    'compute_refine_loss',
]
