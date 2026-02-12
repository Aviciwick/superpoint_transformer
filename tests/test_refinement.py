"""
ResidualRefinementHead 单元测试

测试残差细化头模块的核心功能，包括：
1. 输出形状测试：验证输出张量的形状
2. 残差加和测试：验证 Final = Base + Delta
3. 防御性测试：验证空索引处理
4. 梯度流测试：验证梯度能正确回传
5. 设备兼容性测试：验证 CUDA 支持
"""

import sys
import os

# 添加项目路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'src'))

import pytest
import torch

# 使用 importlib 加载模块
import importlib.util
refinement_path = os.path.join(project_root, 'src', 'Hybrid-SPT', 'ResidualRefinementHead', 'refinement.py')
spec = importlib.util.spec_from_file_location('refinement', refinement_path)
refinement_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(refinement_module)
ResidualRefinementHead = refinement_module.ResidualRefinementHead


class TestResidualRefinementHead:
    """ResidualRefinementHead 测试类"""
    
    @pytest.fixture
    def head(self):
        """创建默认配置的模块实例"""
        return ResidualRefinementHead(
            d_model=64,
            num_classes=13,
            dropout=0.1
        )
    
    @pytest.fixture
    def sample_data(self):
        """生成样本测试数据"""
        torch.manual_seed(42)
        K, N, D, C = 20, 32, 64, 13
        
        enhanced_sp = torch.randn(K, D)
        point_feats = torch.randn(K, N, D)
        coarse_logits = torch.randn(K, C)
        
        return enhanced_sp, point_feats, coarse_logits
    
    # ==========================================
    # 1. 输出形状测试
    # ==========================================
    
    def test_output_shape(self, head, sample_data):
        """测试输出张量的形状是否正确"""
        enhanced, points, coarse = sample_data
        K, N = enhanced.shape[0], points.shape[1]
        C = head.num_classes
        
        point_logits = head(enhanced, points, coarse)
        
        assert point_logits.shape == (K, N, C), \
            f"point_logits 形状错误: 期望 ({K}, {N}, {C}), 实际 {point_logits.shape}"
    
    # ==========================================
    # 2. 残差加和测试
    # ==========================================
    
    def test_residual_addition(self, sample_data):
        """验证残差加和逻辑正确"""
        enhanced, points, coarse = sample_data
        K, N = enhanced.shape[0], points.shape[1]
        
        # 创建一个简单的 head 来测试残差逻辑
        head = ResidualRefinementHead(d_model=64, num_classes=13)
        
        # 设置 MLP 权重为零，这样 delta 应该接近零
        with torch.no_grad():
            for layer in head.refinement_head:
                if hasattr(layer, 'weight'):
                    layer.weight.zero_()
                if hasattr(layer, 'bias'):
                    layer.bias.zero_()
        
        point_logits = head(enhanced, points, coarse)
        
        # 验证输出接近 coarse_logits 广播后的值
        expected = coarse.unsqueeze(1).expand(-1, N, -1)
        assert torch.allclose(point_logits, expected, atol=1e-5), \
            "当 delta=0 时，输出应等于广播后的 coarse_logits"
    
    def test_pure_prediction_mode(self, head, sample_data):
        """验证纯预测模式（不传入 coarse_logits）"""
        enhanced, points, _ = sample_data
        
        # 不传入 coarse_logits
        point_logits = head(enhanced, points)
        
        assert point_logits.shape == (enhanced.shape[0], points.shape[1], head.num_classes)
    
    # ==========================================
    # 3. 防御性测试
    # ==========================================
    
    def test_empty_input(self, head):
        """测试 K=0 的边界情况"""
        enhanced = torch.randn(0, 64)
        points = torch.randn(0, 32, 64)
        coarse = torch.randn(0, 13)
        
        point_logits = head(enhanced, points, coarse)
        
        assert point_logits.numel() == 0, "空输入应返回空张量"
        assert point_logits.shape[0] == 0
    
    def test_single_superpoint(self, head):
        """测试单个超点的情况"""
        torch.manual_seed(42)
        enhanced = torch.randn(1, 64)
        points = torch.randn(1, 32, 64)
        coarse = torch.randn(1, 13)
        
        point_logits = head(enhanced, points, coarse)
        
        assert point_logits.shape == (1, 32, 13)
    
    def test_dimension_mismatch_error(self, head):
        """测试维度不匹配时抛出异常"""
        enhanced = torch.randn(10, 64)
        points = torch.randn(10, 32, 32)  # 维度不匹配
        coarse = torch.randn(10, 13)
        
        with pytest.raises(ValueError, match="维度"):
            head(enhanced, points, coarse)
    
    def test_k_mismatch_error(self, head):
        """测试 K 不匹配时抛出异常"""
        enhanced = torch.randn(10, 64)
        points = torch.randn(10, 32, 64)
        coarse = torch.randn(5, 13)  # K 不匹配
        
        with pytest.raises(ValueError, match="形状不匹配"):
            head(enhanced, points, coarse)
    
    # ==========================================
    # 4. 梯度流测试
    # ==========================================
    
    def test_gradient_flow(self, head, sample_data):
        """验证梯度能正确回传"""
        enhanced, points, coarse = sample_data
        
        # 设置 requires_grad
        enhanced = enhanced.clone().requires_grad_(True)
        points = points.clone().requires_grad_(True)
        coarse = coarse.clone().requires_grad_(True)
        
        point_logits = head(enhanced, points, coarse)
        
        # 对输出求和并反向传播
        loss = point_logits.sum()
        loss.backward()
        
        # 验证梯度存在
        assert enhanced.grad is not None, "enhanced 没有梯度"
        assert points.grad is not None, "points 没有梯度"
        assert coarse.grad is not None, "coarse 没有梯度"
        
        # 验证梯度非零
        assert enhanced.grad.abs().sum() > 0, "enhanced 梯度全为零"
        assert points.grad.abs().sum() > 0, "points 梯度全为零"
        assert coarse.grad.abs().sum() > 0, "coarse 梯度全为零"
    
    # ==========================================
    # 5. 数值稳定性测试
    # ==========================================
    
    def test_no_nan_with_extreme_values(self, head):
        """测试极端输入值不产生 NaN"""
        torch.manual_seed(42)
        enhanced = torch.randn(10, 64) * 100
        points = torch.randn(10, 32, 64) * 100
        coarse = torch.randn(10, 13) * 100
        
        point_logits = head(enhanced, points, coarse)
        
        assert not torch.isnan(point_logits).any(), "输出包含 NaN"
        assert not torch.isinf(point_logits).any(), "输出包含 Inf"
    
    # ==========================================
    # 6. 设备兼容性测试
    # ==========================================
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA 不可用")
    def test_cuda_compatibility(self, head, sample_data):
        """测试 CUDA 设备兼容性"""
        enhanced, points, coarse = sample_data
        device = torch.device("cuda")
        
        # 移动到 CUDA
        head = head.to(device)
        enhanced = enhanced.to(device)
        points = points.to(device)
        coarse = coarse.to(device)
        
        point_logits = head(enhanced, points, coarse)
        
        assert point_logits.device.type == "cuda", "输出设备不匹配"
    
    # ==========================================
    # 7. 初始化验证测试
    # ==========================================
    
    def test_invalid_d_model(self):
        """测试无效的 d_model 参数"""
        with pytest.raises(AssertionError):
            ResidualRefinementHead(d_model=0)
    
    def test_invalid_num_classes(self):
        """测试无效的 num_classes 参数"""
        with pytest.raises(AssertionError):
            ResidualRefinementHead(num_classes=0)
    
    def test_extra_repr(self, head):
        """测试模块字符串表示"""
        repr_str = head.extra_repr()
        assert "d_model" in repr_str
        assert "num_classes" in repr_str


# ==========================================
# 运行测试
# ==========================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
