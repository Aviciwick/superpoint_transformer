"""
AdaptiveUncertaintySampler 单元测试

测试自适应不确定性采样器的核心功能，包括：
1. 基础功能测试：验证输出形状和索引有效性
2. 边界条件测试：验证 K >= 1 保护
3. None 特征回退测试：验证 handcrafted_features=None 时正确回退
4. 数值稳定性测试：验证极端输入不产生 NaN
"""

import sys
import os

# 添加项目路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'src'))

import pytest
import torch

# 由于目录名包含空格和连字符，需要使用 importlib 加载模块
import importlib.util
import math
sampler_path = os.path.join(project_root, 'src', 'Hybrid-SPT', 'AdaptiveUncertaintySampler', 'sampler.py')
spec = importlib.util.spec_from_file_location('sampler', sampler_path)
sampler_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sampler_module)
AdaptiveUncertaintySampler = sampler_module.AdaptiveUncertaintySampler


class TestAdaptiveUncertaintySampler:
    """AdaptiveUncertaintySampler 测试类"""
    
    @pytest.fixture
    def sampler(self):
        """创建默认配置的采样器实例"""
        return AdaptiveUncertaintySampler(
            topk_ratio=0.20,
            alpha=0.7,
            beta=0.3,
            scatter_idx=2
        )
    
    @pytest.fixture
    def sample_data(self):
        """生成样本测试数据"""
        torch.manual_seed(42)
        M, C, D = 100, 20, 9
        coarse_logits = torch.randn(M, C)
        handcrafted_features = torch.rand(M, D)
        return coarse_logits, handcrafted_features
    
    # ==========================================
    # 1. 基础功能测试
    # ==========================================
    
    def test_output_shape(self, sampler, sample_data):
        """
        测试输出形状是否正确。
        
        验证项：
        - hard_sp_indices 形状应为 [K]，K = ceil(M * topk_ratio)
        - debug_scores 形状应为 [M]
        """
        coarse_logits, handcrafted_features = sample_data
        M = coarse_logits.shape[0]
        
        hard_indices, scores = sampler(coarse_logits, handcrafted_features)
        
        # 根据设计文档使用 ceil 计算期望 K
        expected_K = max(1, math.ceil(M * sampler.topk_ratio))
        assert hard_indices.shape == (expected_K,), \
            f"索引形状错误: 期望 ({expected_K},), 实际 {hard_indices.shape}"
        assert scores.shape == (M,), \
            f"评分形状错误: 期望 ({M},), 实际 {scores.shape}"
    
    def test_indices_validity(self, sampler, sample_data):
        """
        测试返回的索引是否有效。
        
        验证项：
        - 所有索引应在 [0, M) 范围内
        - 索引应为整数类型
        """
        coarse_logits, handcrafted_features = sample_data
        M = coarse_logits.shape[0]
        
        hard_indices, _ = sampler(coarse_logits, handcrafted_features)
        
        assert hard_indices.dtype == torch.int64, \
            f"索引类型错误: 期望 torch.int64, 实际 {hard_indices.dtype}"
        assert (hard_indices >= 0).all(), "存在负索引"
        assert (hard_indices < M).all(), f"存在越界索引 (max={hard_indices.max()}, M={M})"
    
    def test_top_indices_are_highest_scores(self, sampler, sample_data):
        """
        测试返回的索引是否对应最高评分。
        
        验证项：
        - 选中超点的最小评分 >= 未选中超点的最大评分
        """
        coarse_logits, handcrafted_features = sample_data
        M = coarse_logits.shape[0]
        
        hard_indices, scores = sampler(coarse_logits, handcrafted_features)
        
        # 创建未选中超点的掩码
        mask = torch.ones(M, dtype=torch.bool)
        mask[hard_indices] = False
        
        selected_min = scores[hard_indices].min()
        unselected_max = scores[mask].max() if mask.any() else selected_min
        
        assert selected_min >= unselected_max, \
            f"Top-K 选择错误: 选中最小分 {selected_min:.4f} < 未选中最大分 {unselected_max:.4f}"
    
    # ==========================================
    # 2. 边界条件测试
    # ==========================================
    
    def test_minimum_k_protection(self):
        """
        测试 K >= 1 边界保护。
        
        验证项：
        - 即使 topk_ratio 非常小，也应至少返回 1 个索引
        """
        sampler = AdaptiveUncertaintySampler(topk_ratio=0.001)  # 非常小的比例
        
        # 只有 5 个超点，按 0.001 比例应该是 0，但应该被保护为 1
        logits = torch.randn(5, 10)
        
        hard_indices, _ = sampler(logits, None)
        
        assert len(hard_indices) >= 1, "K 应至少为 1"
    
    def test_single_superpoint(self, sampler):
        """
        测试单个超点的情况。
        
        验证项：
        - M=1 时应正常工作
        """
        logits = torch.randn(1, 20)
        features = torch.rand(1, 9)
        
        hard_indices, scores = sampler(logits, features)
        
        assert len(hard_indices) == 1
        assert hard_indices[0] == 0
    
    def test_full_sampling(self):
        """
        测试 100% 采样。
        
        验证项：
        - topk_ratio=1.0 时应返回所有超点
        """
        sampler = AdaptiveUncertaintySampler(topk_ratio=1.0)
        M = 50
        logits = torch.randn(M, 20)
        
        hard_indices, _ = sampler(logits, None)
        
        assert len(hard_indices) == M, f"100% 采样应返回所有 {M} 个超点"
    
    # ==========================================
    # 3. None 特征回退测试
    # ==========================================
    
    def test_none_features_fallback(self, sampler, sample_data):
        """
        测试 handcrafted_features=None 时的回退机制。
        
        验证项：
        - 应正常返回结果，不报错
        - 评分应仅基于语义熵
        """
        coarse_logits, _ = sample_data
        
        hard_indices, scores = sampler(coarse_logits, None)
        
        assert hard_indices is not None
        assert scores is not None
        assert not torch.isnan(scores).any(), "存在 NaN 评分"
    
    def test_zero_beta_same_as_none_features(self, sample_data):
        """
        测试 beta=0 时与无几何特征的等价性。
        
        验证项：
        - beta=0 的评分应与 handcrafted_features=None 的评分数值一致（相同 alpha）
        """
        coarse_logits, handcrafted_features = sample_data
        
        # 使用相同的 alpha 确保评分数值一致
        sampler_zero_beta = AdaptiveUncertaintySampler(alpha=0.7, beta=0.0)
        sampler_for_none = AdaptiveUncertaintySampler(alpha=0.7, beta=0.3)
        
        _, scores_zero_beta = sampler_zero_beta(coarse_logits, handcrafted_features)
        _, scores_none = sampler_for_none(coarse_logits, None)
        
        # 两者的评分应数值一致（都只使用熵，且 alpha 相同）
        assert scores_zero_beta.shape == scores_none.shape, "评分形状不一致"
        assert torch.allclose(scores_zero_beta, scores_none, atol=1e-5), \
            f"beta=0 与 features=None 的评分应数值一致，差异: {(scores_zero_beta - scores_none).abs().max():.6f}"
    
    # ==========================================
    # 4. 数值稳定性测试
    # ==========================================
    
    def test_no_nan_with_uniform_logits(self, sampler):
        """
        测试均匀 logits（最大熵）时不产生 NaN。
        
        验证项：
        - 均匀分布的 logits 应产生最大熵但无 NaN
        """
        M, C = 50, 20
        uniform_logits = torch.zeros(M, C)  # 均匀分布
        
        hard_indices, scores = sampler(uniform_logits, None)
        
        assert not torch.isnan(scores).any(), "均匀分布产生 NaN"
        assert not torch.isinf(scores).any(), "均匀分布产生 Inf"
    
    def test_no_nan_with_one_hot_logits(self, sampler):
        """
        测试 one-hot logits（最小熵）时不产生 NaN。
        
        验证项：
        - 完全确定的预测应产生最小熵但无 NaN
        """
        M, C = 50, 20
        one_hot_logits = torch.zeros(M, C)
        one_hot_logits[:, 0] = 100.0  # 强确定性预测
        
        hard_indices, scores = sampler(one_hot_logits, None)
        
        assert not torch.isnan(scores).any(), "One-hot 分布产生 NaN"
        assert not torch.isinf(scores).any(), "One-hot 分布产生 Inf"
    
    def test_no_nan_with_extreme_features(self, sampler, sample_data):
        """
        测试极端几何特征值时不产生 NaN。
        
        验证项：
        - 极大/极小的特征值应正常归一化
        """
        coarse_logits, _ = sample_data
        M = coarse_logits.shape[0]
        
        # 创建极端特征值
        extreme_features = torch.zeros(M, 9)
        extreme_features[:, 2] = torch.linspace(0, 1e6, M)  # 极大范围
        
        hard_indices, scores = sampler(coarse_logits, extreme_features)
        
        assert not torch.isnan(scores).any(), "极端特征值产生 NaN"
        assert not torch.isinf(scores).any(), "极端特征值产生 Inf"
    
    def test_constant_features(self, sampler, sample_data):
        """
        测试常数几何特征（min=max）时不产生 NaN。
        
        验证项：
        - 所有特征值相同时，归一化应处理除零
        """
        coarse_logits, _ = sample_data
        M = coarse_logits.shape[0]
        
        # 创建常数特征
        constant_features = torch.ones(M, 9) * 5.0
        
        hard_indices, scores = sampler(coarse_logits, constant_features)
        
        assert not torch.isnan(scores).any(), "常数特征产生 NaN"
    
    # ==========================================
    # 5. 设备兼容性测试
    # ==========================================
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA 不可用")
    def test_cuda_compatibility(self, sampler):
        """
        测试 CUDA 设备兼容性。
        
        验证项：
        - 在 GPU 上应正常工作
        - 输出应在相同设备上
        """
        device = torch.device("cuda")
        
        logits = torch.randn(100, 20, device=device)
        features = torch.rand(100, 9, device=device)
        
        hard_indices, scores = sampler(logits, features)
        
        # 检查设备类型而非设备对象，因为 torch.device("cuda") != torch.device("cuda:0")
        assert hard_indices.device.type == "cuda", "索引设备不匹配"
        assert scores.device.type == "cuda", "评分设备不匹配"
    
    # ==========================================
    # 6. 初始化验证测试
    # ==========================================
    
    def test_invalid_topk_ratio(self):
        """测试无效的 topk_ratio 参数"""
        with pytest.raises(AssertionError):
            AdaptiveUncertaintySampler(topk_ratio=0.0)
        
        with pytest.raises(AssertionError):
            AdaptiveUncertaintySampler(topk_ratio=1.5)
    
    def test_negative_weights(self):
        """测试负权重参数"""
        with pytest.raises(AssertionError):
            AdaptiveUncertaintySampler(alpha=-0.1)
        
        with pytest.raises(AssertionError):
            AdaptiveUncertaintySampler(beta=-0.1)
    
    def test_extra_repr(self, sampler):
        """测试模块字符串表示"""
        repr_str = sampler.extra_repr()
        assert "topk_ratio" in repr_str
        assert "alpha" in repr_str
        assert "beta" in repr_str
    
    def test_empty_superpoints(self, sampler):
        """测试 M==0 边界情况"""
        empty_logits = torch.randn(0, 20)
        hard_indices, scores = sampler(empty_logits, None)
        
        assert hard_indices.numel() == 0, "空输入应返回空索引"
        assert scores.numel() == 0, "空输入应返回空评分"
    
    def test_handcrafted_features_shape_mismatch(self, sampler):
        """测试 handcrafted_features 形状不匹配时抛出异常"""
        logits = torch.randn(100, 20)
        wrong_features = torch.randn(50, 9)  # 应该是 100
        
        with pytest.raises(ValueError, match="形状不匹配"):
            sampler(logits, wrong_features)
    
    def test_coarse_logits_not_2d(self, sampler):
        """测试 coarse_logits 非 2D 时抛出异常"""
        wrong_logits = torch.randn(100, 20, 5)  # 3D 张量
        
        with pytest.raises(ValueError, match="2D 张量"):
            sampler(wrong_logits, None)
    
    def test_coarse_logits_zero_classes(self, sampler):
        """测试类别数为 0 时抛出异常"""
        zero_class_logits = torch.randn(100, 0)
        
        with pytest.raises(ValueError, match="C 必须 > 0"):
            sampler(zero_class_logits, None)


# ==========================================
# 运行测试
# ==========================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
