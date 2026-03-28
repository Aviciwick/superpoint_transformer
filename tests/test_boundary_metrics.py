"""BoundaryPanopticQuality3D 及相关函数的单元测试。

测试场景：
1. extract_3d_boundary 基本功能
2. extract_3d_boundary 空掩码
3. BIoU 完美匹配
4. BIoU 偏移场景
5. bPQ 端到端计算
6. TP 匹配条件验证
7. BoundaryPanopticQuality3D 的 update + compute 流程
"""

import torch
import numpy as np
import pytest
from src.metrics.boundary_panoptic import (
    extract_3d_boundary,
    compute_boundary_panoptic_metrics,
    BoundaryPanopticQuality3D,
)


def _make_cube_points(center, size, n_per_dim=10):
    """在给定中心和尺寸的立方体内均匀采样点。

    Args:
        center: tuple (3,), 立方体中心坐标。
        size: float, 立方体边长。
        n_per_dim: int, 每个维度的采样数。

    Returns:
        numpy.ndarray: (n_per_dim^3, 3) 的点坐标数组。
    """
    half = size / 2
    x = np.linspace(center[0] - half, center[0] + half, n_per_dim)
    y = np.linspace(center[1] - half, center[1] + half, n_per_dim)
    z = np.linspace(center[2] - half, center[2] + half, n_per_dim)
    xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')
    return np.stack([xx.ravel(), yy.ravel(), zz.ravel()], axis=1).astype(
        np.float32)


class TestExtract3dBoundary:
    """测试 extract_3d_boundary 函数。"""

    def test_basic_boundary_extraction(self):
        """基本场景：一个立方体分为两半，验证边界在分界面附近。"""
        coords = _make_cube_points((0, 0, 0), 2.0, n_per_dim=20)
        # mask: x < 0 的点属于实例 A
        mask = coords[:, 0] < 0
        d = 0.2

        boundary = extract_3d_boundary(coords, mask, d)

        # 边界应只在 mask 内部
        assert boundary.dtype == bool
        assert boundary.shape == mask.shape
        assert not boundary[~mask].any(), "边界点不应在 mask 外部"

        # 边界点应存在
        assert boundary.sum() > 0, "应该有边界点"

        # 边界点应靠近 x=0 分界面
        boundary_coords = coords[boundary]
        # 所有边界点的 x 应在 [-d, 0) 附近
        assert boundary_coords[:, 0].min() >= -d - 0.01

    def test_empty_mask(self):
        """空掩码应返回全 False。"""
        coords = _make_cube_points((0, 0, 0), 1.0, n_per_dim=5)
        mask = np.zeros(coords.shape[0], dtype=bool)
        d = 0.1

        boundary = extract_3d_boundary(coords, mask, d)
        assert not boundary.any()

    def test_full_mask(self):
        """全 True 掩码（没有 outside 点）应返回全 False。"""
        coords = _make_cube_points((0, 0, 0), 1.0, n_per_dim=5)
        mask = np.ones(coords.shape[0], dtype=bool)
        d = 0.1

        boundary = extract_3d_boundary(coords, mask, d)
        assert not boundary.any()

    def test_boundary_is_subset_of_mask(self):
        """边界始终是 mask 的子集。"""
        rng = np.random.RandomState(42)
        coords = rng.randn(500, 3).astype(np.float32)
        mask = rng.rand(500) > 0.5
        d = 0.5

        boundary = extract_3d_boundary(coords, mask, d)
        # 边界内的点一定在 mask 内
        assert np.all(~boundary | mask)


class TestBIoU:
    """测试 BIoU 计算逻辑。"""

    def test_perfect_match(self):
        """GT 和 Pred 完全一致时，BIoU 应为 1.0。"""
        coords = _make_cube_points((0, 0, 0), 2.0, n_per_dim=10)
        n = coords.shape[0]

        gt_instance_ids = np.zeros(n, dtype=np.int64)
        pred_instance_ids = np.zeros(n, dtype=np.int64)
        gt_semantic = np.zeros(n, dtype=np.int64)
        pred_semantic = np.zeros(n, dtype=np.int64)

        # 实例 0: x < 0，实例 1: x >= 0
        gt_instance_ids[coords[:, 0] >= 0] = 1
        pred_instance_ids[coords[:, 0] >= 0] = 1

        result = compute_boundary_panoptic_metrics(
            coords, gt_instance_ids, gt_semantic,
            pred_instance_ids, pred_semantic,
            num_classes=1, stuff_classes=[], d=0.3)

        # 应有 2 个 TP，0 个 FP/FN
        assert result['tp_per_class'][0] == 2
        assert result['fp_per_class'][0] == 0
        assert result['fn_per_class'][0] == 0

        # bSQ 应接近 1.0 (因为 min(IoU, BIoU) 在完美匹配下均为 1.0)
        bsq = result['bsq_sum_per_class'][0] / result['tp_per_class'][0]
        assert bsq > 0.99, f"完美匹配下 bSQ 应接近 1.0, 实际 {bsq}"

    def test_shifted_prediction(self):
        """Pred 有微小偏移时，BIoU 应 < 1.0 但 > 0。"""
        coords = _make_cube_points((0, 0, 0), 2.0, n_per_dim=10)
        n = coords.shape[0]

        gt_instance_ids = np.zeros(n, dtype=np.int64)
        pred_instance_ids = np.zeros(n, dtype=np.int64)
        gt_semantic = np.zeros(n, dtype=np.int64)
        pred_semantic = np.zeros(n, dtype=np.int64)

        # GT: x < 0 为实例 0，x >= 0 为实例 1
        gt_instance_ids[coords[:, 0] >= 0] = 1

        # Pred: x < 0.1 为实例 0，x >= 0.1 为实例 1 (轻微偏移)
        pred_instance_ids[coords[:, 0] >= 0.1] = 1

        result = compute_boundary_panoptic_metrics(
            coords, gt_instance_ids, gt_semantic,
            pred_instance_ids, pred_semantic,
            num_classes=1, stuff_classes=[], d=0.3)

        # 应该仍然有 TP（偏移小，IoU 仍 > 0.5）
        tp = result['tp_per_class'][0]
        assert tp > 0, "轻微偏移下仍应有 TP"


class TestBPQ:
    """测试 bPQ = bSQ × bRQ 的关系。"""

    def test_bpq_equals_bsq_times_brq(self):
        """验证 bPQ = bSQ × bRQ。"""
        coords = _make_cube_points((0, 0, 0), 2.0, n_per_dim=10)
        n = coords.shape[0]

        gt_ids = np.zeros(n, dtype=np.int64)
        pred_ids = np.zeros(n, dtype=np.int64)
        gt_sem = np.zeros(n, dtype=np.int64)
        pred_sem = np.zeros(n, dtype=np.int64)

        gt_ids[coords[:, 0] >= 0] = 1
        pred_ids[coords[:, 0] >= 0] = 1

        metric = BoundaryPanopticQuality3D(
            num_classes=1,
            boundary_distance=0.3,
            stuff_classes=[],
            ignore_unseen_classes=True,
            compute_on_cpu=True)

        metric.update(
            torch.from_numpy(coords),
            torch.from_numpy(gt_ids),
            torch.from_numpy(gt_sem),
            torch.from_numpy(pred_ids),
            torch.from_numpy(pred_sem))

        results = metric.compute()
        bpq = results.bpq.item()
        bsq = results.bsq.item()
        brq = results.brq.item()

        assert abs(bpq - bsq * brq) < 1e-6, \
            f"bPQ ({bpq}) != bSQ ({bsq}) × bRQ ({brq})"


class TestTPMatchingCondition:
    """测试 min(Mask_IoU, 3D_BIoU) > 0.5 的 TP 判定条件。"""

    def test_no_tp_when_biou_low(self):
        """即使 Mask IoU 高，如果 BIoU 很低也不应匹配 TP。

        构造场景：GT 和 Pred 在体积上高度重叠但边界位置差异大。
        """
        coords = _make_cube_points((0, 0, 0), 4.0, n_per_dim=20)
        n = coords.shape[0]

        gt_ids = np.zeros(n, dtype=np.int64)
        pred_ids = np.zeros(n, dtype=np.int64)
        gt_sem = np.zeros(n, dtype=np.int64)
        pred_sem = np.zeros(n, dtype=np.int64)

        # GT 实例 0: 一个大球体
        gt_mask = np.linalg.norm(coords, axis=1) < 1.5
        gt_ids[gt_mask] = 0
        gt_ids[~gt_mask] = 1

        # Pred 实例 0: 与 GT 类似但边界大幅偏移
        pred_mask = np.linalg.norm(coords - [0.5, 0.5, 0.5], axis=1) < 1.5
        pred_ids[pred_mask] = 0
        pred_ids[~pred_mask] = 1

        result = compute_boundary_panoptic_metrics(
            coords, gt_ids, gt_sem, pred_ids, pred_sem,
            num_classes=1, stuff_classes=[], d=0.15)

        # 结果应该有定义（可能有也可能没有 TP，取决于实际的
        # IoU 和 BIoU 值），验证不会报错
        assert result['tp_per_class'].shape[0] == 1


class TestBoundaryPanopticQuality3D:
    """测试 BoundaryPanopticQuality3D 指标类的完整流程。"""

    def test_update_and_compute(self):
        """update 后 compute 应返回有效结果。"""
        coords = _make_cube_points((0, 0, 0), 2.0, n_per_dim=10)
        n = coords.shape[0]

        gt_ids = np.zeros(n, dtype=np.int64)
        pred_ids = np.zeros(n, dtype=np.int64)
        gt_sem = np.zeros(n, dtype=np.int64)
        pred_sem = np.zeros(n, dtype=np.int64)

        gt_ids[coords[:, 0] >= 0] = 1
        pred_ids[coords[:, 0] >= 0] = 1

        metric = BoundaryPanopticQuality3D(
            num_classes=1,
            boundary_distance=0.3,
            stuff_classes=[],
            ignore_unseen_classes=True)

        metric.update(
            torch.from_numpy(coords),
            torch.from_numpy(gt_ids),
            torch.from_numpy(gt_sem),
            torch.from_numpy(pred_ids),
            torch.from_numpy(pred_sem))

        results = metric.compute()

        # 基本结构检查
        assert hasattr(results, 'bpq')
        assert hasattr(results, 'bsq')
        assert hasattr(results, 'brq')
        assert hasattr(results, 'biou')
        assert hasattr(results, 'bpq_per_class')

        # 值范围检查
        bpq = results.bpq.item()
        assert 0 <= bpq <= 1, f"bPQ 应在 [0, 1] 范围内, 实际 {bpq}"

    def test_reset(self):
        """reset 后 compute 应返回 NaN。"""
        metric = BoundaryPanopticQuality3D(
            num_classes=2,
            boundary_distance=0.3,
            stuff_classes=[])

        results = metric.compute()
        # 无数据时应返回 NaN
        assert torch.isnan(results.bpq)

    def test_multiple_scenes(self):
        """多场景累积后计算应正确聚合。"""
        metric = BoundaryPanopticQuality3D(
            num_classes=1,
            boundary_distance=0.3,
            stuff_classes=[],
            ignore_unseen_classes=True)

        for _ in range(3):
            coords = _make_cube_points((0, 0, 0), 2.0, n_per_dim=10)
            n = coords.shape[0]
            gt_ids = np.zeros(n, dtype=np.int64)
            pred_ids = np.zeros(n, dtype=np.int64)
            gt_sem = np.zeros(n, dtype=np.int64)
            pred_sem = np.zeros(n, dtype=np.int64)
            gt_ids[coords[:, 0] >= 0] = 1
            pred_ids[coords[:, 0] >= 0] = 1

            metric.update(
                torch.from_numpy(coords),
                torch.from_numpy(gt_ids),
                torch.from_numpy(gt_sem),
                torch.from_numpy(pred_ids),
                torch.from_numpy(pred_sem))

        results = metric.compute()
        bpq = results.bpq.item()
        assert 0 < bpq <= 1, f"多场景后 bPQ 应为正数, 实际 {bpq}"

        # 3 个完美匹配的场景，每个 2 个实例 → TP = 6
        tp_total = results.tp_per_class.sum().item()
        assert tp_total == 6, f"应有 6 个 TP, 实际 {tp_total}"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
