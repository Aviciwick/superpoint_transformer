"""
H-SPT 端到端冒烟测试

测试 AUS → CAFM → RRH 三模块串联的完整数据流。

验证目标：
1. Shape 一致性：各模块输入输出维度匹配
2. Device 一致性：张量在同一设备上
3. 双分支验证：残差模式 / 纯预测模式
4. 边界情况：K=0 时整个链路不崩溃
5. 梯度流：端到端梯度能正确回传
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

# 加载 AUS
aus_path = os.path.join(project_root, 'src', 'Hybrid-SPT', 'AdaptiveUncertaintySampler', 'sampler.py')
spec = importlib.util.spec_from_file_location('sampler', aus_path)
sampler_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sampler_module)
AdaptiveUncertaintySampler = sampler_module.AdaptiveUncertaintySampler

# 加载 CAFM
cafm_path = os.path.join(project_root, 'src', 'Hybrid-SPT', 'CrossAttentionFusion', 'fusion.py')
spec = importlib.util.spec_from_file_location('fusion', cafm_path)
fusion_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fusion_module)
CrossAttentionFusionModule = fusion_module.CrossAttentionFusionModule

# 加载 RRH
rrh_path = os.path.join(project_root, 'src', 'Hybrid-SPT', 'ResidualRefinementHead', 'refinement.py')
spec = importlib.util.spec_from_file_location('refinement', rrh_path)
refinement_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(refinement_module)
ResidualRefinementHead = refinement_module.ResidualRefinementHead


def run_hspt_pipeline(
    aus, cafm, rrh,
    coarse_logits_all, all_sp_features, all_sp_centroids, packed_raw_points,
    handcrafted_features=None, use_residual=True
):
    """
    H-SPT 三模块串联流水线。
    
    参数:
        aus: AdaptiveUncertaintySampler 实例
        cafm: CrossAttentionFusionModule 实例
        rrh: ResidualRefinementHead 实例
        coarse_logits_all: [M, C] 全部超点 logits
        all_sp_features: [M, D] 全部超点特征
        all_sp_centroids: [M, 3] 全部超点质心
        packed_raw_points: [M, N, d_raw] 预打包点云
        handcrafted_features: [M, D_geo] 可选几何特征
        use_residual: 是否使用残差模式
    
    返回:
        hard_sp_indices: [K]
        fused_global_features: [M, D]
        point_logits: [K, N, C]
    """
    # Module 1: AUS
    hard_sp_indices, scores = aus(coarse_logits_all, handcrafted_features)
    
    # Module 2: CAFM
    enhanced_k, point_feats_k, fused_global = cafm(
        hard_sp_indices, all_sp_features, all_sp_centroids, packed_raw_points
    )
    
    # Module 3: RRH
    if use_residual and hard_sp_indices.numel() > 0:
        coarse_logits_k = coarse_logits_all[hard_sp_indices]
    else:
        coarse_logits_k = None
    
    point_logits = rrh(enhanced_k, point_feats_k, coarse_logits_k)
    
    return hard_sp_indices, fused_global, point_logits


class TestHSPTE2E:
    """H-SPT 端到端测试类"""
    
    @pytest.fixture
    def modules(self):
        """创建三个模块实例"""
        aus = AdaptiveUncertaintySampler(topk_ratio=0.2)
        cafm = CrossAttentionFusionModule(d_model=64, n_heads=4, d_raw=6)
        rrh = ResidualRefinementHead(d_model=64, num_classes=13)
        return aus, cafm, rrh
    
    @pytest.fixture
    def sample_data(self):
        """生成样本测试数据"""
        torch.manual_seed(42)
        
        M = 100   # 超点总数
        N = 32    # 每个超点的采样点数
        D = 64    # 特征维度
        C = 13    # 类别数
        d_raw = 6 # 原始点维度 (XYZ + 其他)
        
        coarse_logits = torch.randn(M, C)
        all_sp_features = torch.randn(M, D)
        all_sp_centroids = torch.randn(M, 3)
        packed_raw_points = torch.randn(M, N, d_raw)
        
        return coarse_logits, all_sp_features, all_sp_centroids, packed_raw_points
    
    # ==========================================
    # 1. Shape 一致性测试
    # ==========================================
    
    def test_pipeline_output_shapes(self, modules, sample_data):
        """验证端到端输出形状"""
        aus, cafm, rrh = modules
        coarse_logits, features, centroids, points = sample_data
        
        M, C = coarse_logits.shape
        N = points.shape[1]
        D = features.shape[1]
        
        hard_indices, fused, point_logits = run_hspt_pipeline(
            aus, cafm, rrh,
            coarse_logits, features, centroids, points,
            use_residual=True
        )
        
        K = hard_indices.numel()
        
        # 验证形状
        assert hard_indices.dim() == 1
        assert fused.shape == (M, D)
        assert point_logits.shape == (K, N, C)
    
    # ==========================================
    # 2. 残差模式 vs 纯预测模式
    # ==========================================
    
    def test_residual_mode(self, modules, sample_data):
        """验证残差模式"""
        aus, cafm, rrh = modules
        coarse_logits, features, centroids, points = sample_data
        
        hard_indices, fused, point_logits = run_hspt_pipeline(
            aus, cafm, rrh,
            coarse_logits, features, centroids, points,
            use_residual=True
        )
        
        assert point_logits is not None
        assert not torch.isnan(point_logits).any()
    
    def test_pure_prediction_mode(self, modules, sample_data):
        """验证纯预测模式（不传入 coarse_logits）"""
        aus, cafm, rrh = modules
        coarse_logits, features, centroids, points = sample_data
        
        hard_indices, fused, point_logits = run_hspt_pipeline(
            aus, cafm, rrh,
            coarse_logits, features, centroids, points,
            use_residual=False
        )
        
        assert point_logits is not None
        assert not torch.isnan(point_logits).any()
    
    # ==========================================
    # 3. 边界情况测试
    # ==========================================
    
    def test_pipeline_with_empty_input(self, modules):
        """验证 M=0 边界情况"""
        aus, cafm, rrh = modules
        
        # 空输入
        coarse_logits = torch.randn(0, 13)
        features = torch.randn(0, 64)
        centroids = torch.randn(0, 3)
        points = torch.randn(0, 32, 6)
        
        hard_indices, fused, point_logits = run_hspt_pipeline(
            aus, cafm, rrh,
            coarse_logits, features, centroids, points,
            use_residual=True
        )
        
        assert hard_indices.numel() == 0
        assert fused.shape[0] == 0
        assert point_logits.numel() == 0
    
    def test_pipeline_with_high_certainty(self, modules, sample_data):
        """验证所有超点都很确定时（可能 K=0 或很小）"""
        aus, cafm, rrh = modules
        coarse_logits, features, centroids, points = sample_data
        
        # 创建高确定性 logits (one-hot)
        M = coarse_logits.shape[0]
        certain_logits = torch.zeros_like(coarse_logits)
        certain_logits[:, 0] = 100.0  # 所有超点都高度确信类别0
        
        hard_indices, fused, point_logits = run_hspt_pipeline(
            aus, cafm, rrh,
            certain_logits, features, centroids, points,
            use_residual=True
        )
        
        # 即使 K 很小，也不应崩溃
        assert fused.shape[0] == M
    
    # ==========================================
    # 4. 梯度流测试
    # ==========================================
    
    def test_end_to_end_gradient_flow(self, modules, sample_data):
        """验证端到端梯度能正确回传"""
        aus, cafm, rrh = modules
        coarse_logits, features, centroids, points = sample_data
        
        # 设置 requires_grad
        features = features.clone().requires_grad_(True)
        points = points.clone().requires_grad_(True)
        
        hard_indices, fused, point_logits = run_hspt_pipeline(
            aus, cafm, rrh,
            coarse_logits, features, centroids, points,
            use_residual=True
        )
        
        # 反向传播
        loss = point_logits.sum()
        loss.backward()
        
        # 验证梯度存在（非 detach 的部分）
        assert features.grad is not None
        assert points.grad is not None
    
    # ==========================================
    # 5. Device 一致性测试
    # ==========================================
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA 不可用")
    def test_cuda_pipeline(self, modules, sample_data):
        """验证 CUDA 设备上的端到端流水线"""
        aus, cafm, rrh = modules
        coarse_logits, features, centroids, points = sample_data
        
        device = torch.device("cuda")
        
        # 移动到 CUDA
        cafm = cafm.to(device)
        rrh = rrh.to(device)
        coarse_logits = coarse_logits.to(device)
        features = features.to(device)
        centroids = centroids.to(device)
        points = points.to(device)
        
        hard_indices, fused, point_logits = run_hspt_pipeline(
            aus, cafm, rrh,
            coarse_logits, features, centroids, points,
            use_residual=True
        )
        
        # 验证输出在 CUDA 上
        assert point_logits.device.type == "cuda"
        assert fused.device.type == "cuda"
    
    # ==========================================
    # 6. 数值稳定性测试
    # ==========================================
    
    def test_no_nan_in_pipeline(self, modules, sample_data):
        """验证整个流水线无 NaN/Inf"""
        aus, cafm, rrh = modules
        coarse_logits, features, centroids, points = sample_data
        
        hard_indices, fused, point_logits = run_hspt_pipeline(
            aus, cafm, rrh,
            coarse_logits, features, centroids, points,
            use_residual=True
        )
        
        assert not torch.isnan(fused).any()
        assert not torch.isinf(fused).any()
        assert not torch.isnan(point_logits).any()
        assert not torch.isinf(point_logits).any()


# ==========================================
# 运行测试
# ==========================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
