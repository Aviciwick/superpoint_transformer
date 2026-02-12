"""
Hybrid-SPT 模块包

包含 H-SPT (Hybrid-Superpoint Transformer) 架构的核心组件：
- Adaptive Uncertainty Sampler (自适应不确定性采样器)
- Cross-Attention Fusion (待实现)
- Residual Refinement Head (待实现)
"""

from .sampler import AdaptiveUncertaintySampler

__all__ = ['AdaptiveUncertaintySampler']
