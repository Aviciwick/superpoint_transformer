"""
H-SPT 集成级 Smoke 测试

测试 hspt 包导入和 HSPTModule 创建/初始化是否正常工作。
"""

import pytest
import torch


class TestHSPTIntegration:
    """集成级测试"""
    
    def test_hspt_import(self):
        """测试 hspt 包所有导出符号是否可用"""
        from src.hspt import (
            AdaptiveUncertaintySampler,
            CrossAttentionFusionModule,
            ResidualRefinementHead,
            build_packed_points,
            run_hspt,
            HSPTOutput,
            HSPTModule,
            compute_refine_loss
        )
        
        # 确保所有类/函数都可用
        assert AdaptiveUncertaintySampler is not None
        assert CrossAttentionFusionModule is not None
        assert ResidualRefinementHead is not None
        assert build_packed_points is not None
        assert run_hspt is not None
        assert HSPTOutput is not None
        assert HSPTModule is not None
        assert compute_refine_loss is not None
    
    def test_hspt_module_creation(self):
        """测试 HSPTModule 创建"""
        from src.hspt import HSPTModule
        
        module = HSPTModule(
            d_model=64,
            num_classes=13,
            topk_ratio=0.2,
            n_sample=32,
            d_raw=3
        )
        
        assert module is not None
        assert module.d_model == 64
        assert module.num_classes == 13
        assert module.n_sample == 32
        
        # 检查子模块
        assert hasattr(module, 'aus')
        assert hasattr(module, 'cafm')
        assert hasattr(module, 'rrh')
    
    def test_compute_refine_loss(self):
        """测试 refine loss 计算"""
        from src.hspt import compute_refine_loss
        
        K, N, C = 5, 32, 13
        device = torch.device('cpu')
        
        point_logits = torch.randn(K, N, C, device=device)
        packed_point_idx = torch.randint(0, 100, (K, N), device=device)
        hard_sp_indices = torch.arange(K, device=device)
        gt_labels = torch.randint(0, C, (200,), device=device)
        packed_mask = torch.ones(K, N, dtype=torch.bool, device=device)
        
        loss = compute_refine_loss(
            point_logits=point_logits,
            packed_point_idx=packed_point_idx,
            hard_sp_indices=hard_sp_indices,
            gt_labels=gt_labels,
            packed_mask=packed_mask,
            num_classes=C,
            ignore_index=C
        )
        
        assert loss is not None
        assert loss.numel() == 1
        assert not torch.isnan(loss)
        assert loss >= 0
    
    def test_compute_refine_loss_empty(self):
        """测试空输入时 refine loss 返回 0"""
        from src.hspt import compute_refine_loss
        
        K, N, C = 0, 32, 13
        device = torch.device('cpu')
        
        point_logits = torch.zeros(K, N, C, device=device)
        packed_point_idx = torch.zeros(K, N, dtype=torch.long, device=device)
        hard_sp_indices = torch.zeros(K, dtype=torch.long, device=device)
        gt_labels = torch.zeros(100, dtype=torch.long, device=device)
        
        loss = compute_refine_loss(
            point_logits=point_logits,
            packed_point_idx=packed_point_idx,
            hard_sp_indices=hard_sp_indices,
            gt_labels=gt_labels,
            num_classes=C
        )
        
        assert loss == 0.0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
