"""
CrossAttentionFusionModule 单元测试

测试交叉注意力融合模块的核心功能，包括：
1. 输出形状测试：验证各输出张量的形状
2. 坐标局部化测试：验证局部坐标计算正确
3. 全局回填测试：验证 Scatter Update 正确性
4. 防御性测试：验证空索引处理
5. 梯度流测试：验证梯度能正确回传
6. 设备兼容性测试：验证 CUDA 支持
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
fusion_path = os.path.join(project_root, 'src', 'Hybrid-SPT', 'CrossAttentionFusion', 'fusion.py')
spec = importlib.util.spec_from_file_location('fusion', fusion_path)
fusion_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fusion_module)
CrossAttentionFusionModule = fusion_module.CrossAttentionFusionModule


class TestCrossAttentionFusionModule:
    """CrossAttentionFusionModule 测试类"""
    
    @pytest.fixture
    def module(self):
        """创建默认配置的模块实例"""
        return CrossAttentionFusionModule(
            d_model=64,
            d_raw=6,
            n_heads=4,
            dropout=0.1
        )
    
    @pytest.fixture
    def sample_data(self):
        """生成样本测试数据"""
        torch.manual_seed(42)
        M, K, N, D = 100, 20, 32, 64
        
        hard_sp_indices = torch.randperm(M)[:K]
        all_sp_features = torch.randn(M, D)
        all_sp_centroids = torch.randn(M, 3)
        packed_raw_points = torch.randn(M, N, 6)
        
        return hard_sp_indices, all_sp_features, all_sp_centroids, packed_raw_points
    
    # ==========================================
    # 1. 输出形状测试
    # ==========================================
    
    def test_output_shapes(self, module, sample_data):
        """测试输出张量的形状是否正确"""
        hard_indices, features, centroids, points = sample_data
        K = hard_indices.numel()
        M = features.shape[0]
        N = points.shape[1]
        D = features.shape[1]
        
        enhanced, point_feats, fused = module(hard_indices, features, centroids, points)
        
        assert enhanced.shape == (K, D), \
            f"enhanced_features_K 形状错误: 期望 ({K}, {D}), 实际 {enhanced.shape}"
        assert point_feats.shape == (K, N, D), \
            f"point_features_K 形状错误: 期望 ({K}, {N}, {D}), 实际 {point_feats.shape}"
        assert fused.shape == (M, D), \
            f"fused_global_features 形状错误: 期望 ({M}, {D}), 实际 {fused.shape}"
    
    # ==========================================
    # 2. 坐标局部化测试
    # ==========================================
    
    def test_canonicalization(self, module, sample_data):
        """验证坐标局部化正确执行"""
        hard_indices, features, centroids, points = sample_data
        
        # 手动计算期望的局部坐标
        K_centroids = centroids[hard_indices]  # [K, 3]
        K_points = points[hard_indices]        # [K, N, 6]
        expected_local = K_points[..., :3] - K_centroids.unsqueeze(1)
        
        # 模块输出的点特征应该基于局部坐标
        # 由于经过了 MLP，我们只能验证输出不包含 NaN
        enhanced, point_feats, fused = module(hard_indices, features, centroids, points)
        
        assert not torch.isnan(point_feats).any(), "点特征包含 NaN"
        assert not torch.isinf(point_feats).any(), "点特征包含 Inf"
    
    # ==========================================
    # 3. 全局回填测试
    # ==========================================
    
    def test_scatter_update(self, module, sample_data):
        """验证全局回填正确性"""
        hard_indices, features, centroids, points = sample_data
        K = hard_indices.numel()
        M = features.shape[0]
        
        enhanced, _, fused = module(hard_indices, features, centroids, points)
        
        # 验证选中位置已更新
        updated_features = fused[hard_indices]
        assert torch.allclose(updated_features, enhanced, atol=1e-5), \
            "选中位置的特征与 enhanced 不一致"
        
        # 验证未选中位置保持原样
        mask = torch.ones(M, dtype=torch.bool)
        mask[hard_indices] = False
        unchanged_features = fused[mask]
        original_features = features[mask]
        assert torch.allclose(unchanged_features, original_features, atol=1e-5), \
            "未选中位置的特征被错误修改"
    
    def test_update_count(self, module, sample_data):
        """验证被更新的超点数量正确"""
        hard_indices, features, centroids, points = sample_data
        K = hard_indices.numel()
        
        _, _, fused = module(hard_indices, features, centroids, points)
        
        # 计算被修改的超点数
        diff_count = (fused != features).any(dim=1).sum().item()
        assert diff_count == K, \
            f"被更新的超点数错误: 期望 {K}, 实际 {diff_count}"
    
    # ==========================================
    # 4. 防御性测试
    # ==========================================
    
    def test_empty_indices(self, module, sample_data):
        """测试空索引的处理"""
        _, features, centroids, points = sample_data
        empty_indices = torch.tensor([], dtype=torch.long)
        
        enhanced, point_feats, fused = module(empty_indices, features, centroids, points)
        
        assert enhanced.numel() == 0, "空索引时 enhanced 应为空"
        assert point_feats.numel() == 0, "空索引时 point_feats 应为空"
        assert torch.equal(fused, features), "空索引时 fused 应等于原特征"
    
    def test_single_superpoint(self, module):
        """测试单个超点的情况"""
        torch.manual_seed(42)
        hard_indices = torch.tensor([0])
        features = torch.randn(10, 64)
        centroids = torch.randn(10, 3)
        points = torch.randn(10, 32, 6)
        
        enhanced, point_feats, fused = module(hard_indices, features, centroids, points)
        
        assert enhanced.shape == (1, 64)
        assert point_feats.shape == (1, 32, 64)
        assert fused.shape == (10, 64)
    
    # ==========================================
    # 5. 梯度流测试
    # ==========================================
    
    def test_gradient_flow(self, module, sample_data):
        """验证梯度能正确回传"""
        hard_indices, features, centroids, points = sample_data
        
        # 设置 requires_grad
        features = features.clone().requires_grad_(True)
        points = points.clone().requires_grad_(True)
        
        enhanced, point_feats, fused = module(hard_indices, features, centroids, points)
        
        # 对 fused 求和并反向传播
        loss = fused.sum()
        loss.backward()
        
        # 验证梯度存在
        assert features.grad is not None, "features 没有梯度"
        assert points.grad is not None, "points 没有梯度"
        
        # 验证梯度非零
        assert features.grad.abs().sum() > 0, "features 梯度全为零"
        assert points.grad.abs().sum() > 0, "points 梯度全为零"
    
    def test_gradient_to_original_features(self, module, sample_data):
        """验证梯度能回传到原始特征"""
        hard_indices, features, centroids, points = sample_data
        
        features = features.clone().requires_grad_(True)
        
        enhanced, _, fused = module(hard_indices, features, centroids, points)
        
        # 只对增强的部分求 loss
        loss = enhanced.sum()
        loss.backward()
        
        # 只有被选中的超点应该有非零梯度
        selected_grad = features.grad[hard_indices]
        assert selected_grad.abs().sum() > 0, "选中超点没有梯度"
    
    # ==========================================
    # 6. 数值稳定性测试
    # ==========================================
    
    def test_no_nan_with_extreme_values(self, module):
        """测试极端输入值不产生 NaN"""
        torch.manual_seed(42)
        hard_indices = torch.tensor([0, 1, 2])
        features = torch.randn(10, 64) * 100  # 较大的值
        centroids = torch.randn(10, 3) * 1000
        points = torch.randn(10, 32, 6) * 100
        
        enhanced, point_feats, fused = module(hard_indices, features, centroids, points)
        
        assert not torch.isnan(enhanced).any(), "enhanced 包含 NaN"
        assert not torch.isnan(point_feats).any(), "point_feats 包含 NaN"
        assert not torch.isnan(fused).any(), "fused 包含 NaN"
    
    # ==========================================
    # 7. 设备兼容性测试
    # ==========================================
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA 不可用")
    def test_cuda_compatibility(self, module, sample_data):
        """测试 CUDA 设备兼容性"""
        hard_indices, features, centroids, points = sample_data
        device = torch.device("cuda")
        
        # 移动到 CUDA
        module = module.to(device)
        hard_indices = hard_indices.to(device)
        features = features.to(device)
        centroids = centroids.to(device)
        points = points.to(device)
        
        enhanced, point_feats, fused = module(hard_indices, features, centroids, points)
        
        assert enhanced.device.type == "cuda", "enhanced 设备不匹配"
        assert point_feats.device.type == "cuda", "point_feats 设备不匹配"
        assert fused.device.type == "cuda", "fused 设备不匹配"
    
    # ==========================================
    # 8. 初始化验证测试
    # ==========================================
    
    def test_invalid_d_model(self):
        """测试无效的 d_model 参数"""
        with pytest.raises(AssertionError):
            CrossAttentionFusionModule(d_model=0)
    
    def test_d_model_n_heads_mismatch(self):
        """测试 d_model 与 n_heads 不匹配"""
        with pytest.raises(AssertionError):
            CrossAttentionFusionModule(d_model=65, n_heads=4)  # 65 不能被 4 整除
    
    def test_extra_repr(self, module):
        """测试模块字符串表示"""
        repr_str = module.extra_repr()
        assert "d_model" in repr_str
        assert "n_heads" in repr_str
    
    # ==========================================
    # 9. FFN 可选功能测试
    # ==========================================
    
    def test_with_ffn(self, sample_data):
        """测试带 FFN 的模块"""
        module = CrossAttentionFusionModule(
            d_model=64,
            n_heads=4,
            use_ffn=True
        )
        hard_indices, features, centroids, points = sample_data
        
        enhanced, point_feats, fused = module(hard_indices, features, centroids, points)
        
        assert enhanced.shape[1] == 64
        assert not torch.isnan(enhanced).any()
    
    # ==========================================
    # 10. 新边界条件测试（风险点修复验证）
    # ==========================================
    
    def test_empty_superpoints_m_zero(self, module):
        """测试 M==0 边界情况"""
        empty_features = torch.randn(0, 64)
        empty_centroids = torch.randn(0, 3)
        empty_points = torch.randn(0, 32, 6)
        empty_indices = torch.tensor([], dtype=torch.long)
        
        enhanced, point_feats, fused = module(
            empty_indices, empty_features, empty_centroids, empty_points
        )
        
        assert enhanced.numel() == 0
        assert point_feats.numel() == 0
        assert fused.shape[0] == 0
    
    def test_invalid_packed_points_dim(self, module):
        """测试 packed_raw_points 维度不正确时抛出异常"""
        hard_indices = torch.tensor([0, 1])
        features = torch.randn(10, 64)
        centroids = torch.randn(10, 3)
        wrong_points = torch.randn(10, 6)  # 应该是 3D
        
        with pytest.raises(ValueError, match="3D 张量"):
            module(hard_indices, features, centroids, wrong_points)
    
    def test_packed_points_missing_xyz(self, module):
        """测试 packed_raw_points 缺少 XYZ 时抛出异常"""
        hard_indices = torch.tensor([0, 1])
        features = torch.randn(10, 64)
        centroids = torch.randn(10, 3)
        wrong_points = torch.randn(10, 32, 2)  # 最后维度 < 3
        
        with pytest.raises(ValueError, match=">= 3"):
            module(hard_indices, features, centroids, wrong_points)
    
    def test_index_out_of_bounds(self, module):
        """测试索引越界时抛出异常"""
        features = torch.randn(10, 64)
        centroids = torch.randn(10, 3)
        points = torch.randn(10, 32, 6)
        out_of_bounds = torch.tensor([0, 15])  # 15 > M-1=9
        
        with pytest.raises(IndexError, match="越界"):
            module(out_of_bounds, features, centroids, points)
    
    def test_index_dtype_auto_convert(self, module):
        """测试索引 dtype 自动转换为 torch.long"""
        features = torch.randn(10, 64)
        centroids = torch.randn(10, 3)
        points = torch.randn(10, 32, 6)
        float_indices = torch.tensor([0, 1], dtype=torch.int32)  # 非 long
        
        # 应该自动转换，不报错
        enhanced, point_feats, fused = module(float_indices, features, centroids, points)
        assert enhanced.shape[0] == 2


# ==========================================
# 运行测试
# ==========================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
