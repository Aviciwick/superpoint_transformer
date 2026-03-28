"""3D Boundary Panoptic Quality 评估指标模块。

本模块实现了用于 3D 点云全景分割的边界评估指标，包括：
- 3D Boundary IoU (3D BIoU)
- 3D Boundary PQ (3D bPQ) = bSQ × bRQ
- 3D Boundary SQ (3D bSQ)
- 3D Boundary RQ (3D bRQ)

核心思想：在实例边界带（距离参数 d 内的区域）上评估分割质量，
用于衡量模型对实例边界的预测精度。

TP 匹配条件升级为 min(Mask_IoU, 3D_BIoU) > 0.5，
严防模型预测空心伪边界的作弊行为。
"""

import torch
import numpy as np
import logging
from typing import Any, List, Optional, Tuple
from collections import namedtuple
from torchmetrics.metric import Metric


log = logging.getLogger(__name__)


SceneData = namedtuple(
    'SceneData',
    ['point_coords', 'gt_instance_ids', 'gt_semantic',
     'pred_instance_ids', 'pred_semantic'])


__all__ = ['BoundaryPanopticQuality3D', 'BoundaryPanopticMetricResults']


class BoundaryPanopticMetricResults:
    """封装 Boundary Panoptic Segmentation 的最终指标结果。

    简单的数据容器类，包含全局指标和逐类别指标。
    不依赖 BaseMetricResults 以避免循环导入。
    """
    __slots__ = (
        'bpq',
        'bsq',
        'brq',
        'biou',
        'bpq_thing',
        'bsq_thing',
        'brq_thing',
        'bpq_stuff',
        'bsq_stuff',
        'brq_stuff',
        'bpq_per_class',
        'bsq_per_class',
        'brq_per_class',
        'biou_per_class',
        'tp_per_class',
        'fp_per_class',
        'fn_per_class',
    )


def extract_3d_boundary(point_coords, mask, d):
    """提取 3D 实例的边界带区域。

    对于给定实例掩码 mask，其边界点定义为：该实例内部，
    距离"非该实例区域"的点在欧氏距离 d 以内的所有点。

    性能优化：先利用 BBox 裁剪到局部区域，再构建 KD-Tree 查询。
    严禁在整个场景上构建 KD-Tree。

    Args:
        point_coords: numpy.ndarray, shape (N, 3)
            所有点的坐标。
        mask: numpy.ndarray, shape (N,), dtype=bool
            当前实例的布尔掩码。
        d: float
            边界带宽度（欧氏距离阈值）。

    Returns:
        numpy.ndarray, shape (N,), dtype=bool
            边界带的布尔掩码（仅 mask 内的点可能为 True）。
    """
    from scipy.spatial import cKDTree

    boundary = np.zeros(mask.shape[0], dtype=bool)

    # 快速退出：实例为空或覆盖全部点（无边界可言）
    inside_idx = np.where(mask)[0]
    outside_idx = np.where(~mask)[0]
    if inside_idx.size == 0 or outside_idx.size == 0:
        return boundary

    # 计算实例 BBox，扩展 d
    inside_coords = point_coords[inside_idx]
    bbox_min = inside_coords.min(axis=0) - d
    bbox_max = inside_coords.max(axis=0) + d

    # 裁剪 outside 点到局部区域（仅保留 BBox 内的 non-mask 点）
    outside_coords = point_coords[outside_idx]
    local_mask = np.all(
        (outside_coords >= bbox_min) & (outside_coords <= bbox_max), axis=1)
    local_outside_coords = outside_coords[local_mask]

    # 如果局部区域内没有 outside 点，则无边界
    if local_outside_coords.shape[0] == 0:
        return boundary

    # 在局部 outside 点上构建 KD-Tree
    tree = cKDTree(local_outside_coords)

    # 查询 inside 点到最近 outside 点的距离
    dists, _ = tree.query(inside_coords, k=1)

    # 距离 < d 的 inside 点即为边界点
    is_boundary = dists < d
    boundary[inside_idx[is_boundary]] = True

    return boundary


def compute_boundary_panoptic_metrics(
    point_coords,
    gt_instance_ids,
    gt_semantic,
    pred_instance_ids,
    pred_semantic,
    num_classes,
    stuff_classes,
    d,
    ignore_unseen_classes=True,
):
    """计算 3D Boundary Panoptic 指标（单场景）。

    核心流水线，复制自原有 PQ 计算逻辑并注入 boundary 条件。

    实现步骤：
    1. 遍历每个语义类别
    2. 对匹配的 GT-Pred 实例对计算 Mask_IoU 和 3D_BIoU
    3. TP 条件：语义匹配 AND min(Mask_IoU, 3D_BIoU) > 0.5
    4. 累加 bSQ = sum(min(Mask_IoU, 3D_BIoU)) / |TP|
    5. bRQ = |TP| / (|TP| + 0.5|FP| + 0.5|FN|)
    6. bPQ = bSQ × bRQ

    Args:
        point_coords: numpy.ndarray, shape (N, 3)
            点坐标。
        gt_instance_ids: numpy.ndarray, shape (N,), dtype=int
            真值实例 ID（-1 表示 void）。
        gt_semantic: numpy.ndarray, shape (N,), dtype=int
            真值语义标签。
        pred_instance_ids: numpy.ndarray, shape (N,), dtype=int
            预测实例 ID。
        pred_semantic: numpy.ndarray, shape (N,), dtype=int
            预测语义标签。
        num_classes: int
            有效类别数。
        stuff_classes: list[int]
            stuff 类别列表。
        d: float
            边界距离参数。
        ignore_unseen_classes: bool
            若为 True，未出现的类别不参与均值计算。

    Returns:
        dict: 包含逐类别和全局的 bpq, bsq, brq, biou, tp, fp, fn。
    """
    # 初始化逐类别累加器
    tp_per_class = np.zeros(num_classes, dtype=np.float64)
    fp_per_class = np.zeros(num_classes, dtype=np.float64)
    fn_per_class = np.zeros(num_classes, dtype=np.float64)
    bsq_sum_per_class = np.zeros(num_classes, dtype=np.float64)
    biou_sum_per_class = np.zeros(num_classes, dtype=np.float64)

    # 标记有效区域（非 void 的 GT 点）
    valid_gt = (gt_semantic >= 0) & (gt_semantic < num_classes)

    # 预计算所有 GT 和 Pred 实例的边界带掩码（缓存以避免重复计算）
    gt_boundary_cache = {}
    pred_boundary_cache = {}

    for cls_id in range(num_classes):
        # 当前类别的 GT 实例和 Pred 实例
        cls_gt_mask = valid_gt & (gt_semantic == cls_id)
        cls_pred_mask = (pred_semantic == cls_id)

        if not cls_gt_mask.any() and not cls_pred_mask.any():
            continue

        # 获取当前类别下的唯一实例 ID
        gt_ids_in_cls = np.unique(gt_instance_ids[cls_gt_mask])
        pred_ids_in_cls = np.unique(pred_instance_ids[cls_pred_mask])

        # 过滤掉 void GT 实例 (id < 0)
        gt_ids_in_cls = gt_ids_in_cls[gt_ids_in_cls >= 0]

        # 构建 IoU 矩阵，大小为 [num_gt_instances, num_pred_instances]
        # 同时计算 BIoU 矩阵
        num_gt = len(gt_ids_in_cls)
        num_pred = len(pred_ids_in_cls)

        if num_gt == 0 and num_pred == 0:
            continue

        # 如果没有 GT 实例，所有 pred 都是 FP
        if num_gt == 0:
            fp_per_class[cls_id] += num_pred
            continue

        # 如果没有 pred 实例，所有 gt 都是 FN
        if num_pred == 0:
            fn_per_class[cls_id] += num_gt
            continue

        # 计算 Mask IoU 和 3D BIoU 矩阵
        mask_iou_matrix = np.zeros((num_gt, num_pred), dtype=np.float64)
        biou_matrix = np.zeros((num_gt, num_pred), dtype=np.float64)

        for gi, gt_id in enumerate(gt_ids_in_cls):
            gt_mask = (gt_instance_ids == gt_id) & cls_gt_mask

            # 提取/缓存 GT 边界
            if gt_id not in gt_boundary_cache:
                gt_boundary_cache[gt_id] = extract_3d_boundary(
                    point_coords, gt_mask, d)
            gt_boundary = gt_boundary_cache[gt_id]

            gt_count = gt_mask.sum()
            gt_bd_count = gt_boundary.sum()

            for pi, pred_id in enumerate(pred_ids_in_cls):
                pred_mask = (pred_instance_ids == pred_id) & cls_pred_mask

                # Mask IoU
                intersection = (gt_mask & pred_mask).sum()
                union = (gt_mask | pred_mask).sum()
                mask_iou = intersection / union if union > 0 else 0.0
                mask_iou_matrix[gi, pi] = mask_iou

                # 3D BIoU
                if gt_bd_count == 0:
                    # GT 无边界（整个实例在最内部），BIoU 退化为 0
                    biou_matrix[gi, pi] = 0.0
                    continue

                # 提取/缓存 Pred 边界
                if pred_id not in pred_boundary_cache:
                    pred_boundary_cache[pred_id] = extract_3d_boundary(
                        point_coords, pred_mask, d)
                pred_boundary = pred_boundary_cache[pred_id]

                # BIoU = |(G_d ∩ G) ∩ (P_d ∩ P)| / |(G_d ∩ G) ∪ (P_d ∩ P)|
                # G_d ∩ G = gt_boundary (已在 gt_mask 内)
                # P_d ∩ P = pred_boundary (已在 pred_mask 内)
                bd_intersection = (gt_boundary & pred_boundary).sum()
                bd_union = (gt_boundary | pred_boundary).sum()
                biou = bd_intersection / bd_union if bd_union > 0 else 0.0
                biou_matrix[gi, pi] = biou

        # TP 匹配：贪心策略（与标准 PQ 一致）
        # min(Mask_IoU, 3D_BIoU) > 0.5
        combined_score = np.minimum(mask_iou_matrix, biou_matrix)

        matched_gt = set()
        matched_pred = set()

        # 按 combined_score 降序贪心匹配
        while True:
            # 找未匹配的最大值
            remaining = combined_score.copy()
            for gi in matched_gt:
                remaining[gi, :] = -1
            for pi in matched_pred:
                remaining[:, pi] = -1

            max_val = remaining.max()
            if max_val <= 0.5:
                break

            gi, pi = np.unravel_index(remaining.argmax(), remaining.shape)
            matched_gt.add(gi)
            matched_pred.add(pi)

            tp_per_class[cls_id] += 1
            bsq_sum_per_class[cls_id] += combined_score[gi, pi]
            biou_sum_per_class[cls_id] += biou_matrix[gi, pi]

        # 未匹配的 GT 为 FN，未匹配的 Pred 为 FP
        fn_per_class[cls_id] += num_gt - len(matched_gt)
        fp_per_class[cls_id] += num_pred - len(matched_pred)

    return {
        'tp_per_class': tp_per_class,
        'fp_per_class': fp_per_class,
        'fn_per_class': fn_per_class,
        'bsq_sum_per_class': bsq_sum_per_class,
        'biou_sum_per_class': biou_sum_per_class,
    }


class BoundaryPanopticQuality3D(Metric):
    """3D Boundary Panoptic Quality 评估指标。

    TorchMetrics 兼容的指标类，用于在多个场景上累积并计算
    3D Boundary PQ/SQ/RQ/BIoU 指标。

    仅在 val/test 阶段使用，不参与训练。

    计算流程：
    1. update() 接收每个场景的点级数据（坐标、实例 ID、语义标签）
    2. compute() 在所有累积场景上计算 boundary panoptic 指标

    Args:
        num_classes: int
            有效类别数。y ∈ [0, num_classes-1] 为有效标签。
        boundary_distance: float
            边界带宽度（默认 0.1，单位与坐标一致）。
        stuff_classes: list[int], optional
            stuff 类别列表。
        ignore_unseen_classes: bool
            若 True，未出现的类别不参与均值计算。
        compute_on_cpu: bool
            若 True，数据存储和计算在 CPU 上进行。
        **kwargs:
            传递给 torchmetrics.Metric 的额外参数。
    """
    full_state_update: bool = False

    def __init__(
            self,
            num_classes: int,
            boundary_distance: float = 0.1,
            stuff_classes: Optional[List[int]] = None,
            ignore_unseen_classes: bool = True,
            compute_on_cpu: bool = True,
            **kwargs: Any
    ) -> None:
        self._compute_on_cpu = compute_on_cpu
        super().__init__(**kwargs)

        self.num_classes = num_classes
        self.boundary_distance = boundary_distance
        self.stuff_classes = stuff_classes or []
        self.ignore_unseen_classes = ignore_unseen_classes

        # 内部状态：按场景累积的数据列表
        # 注意：不使用 add_state，因为 numpy 数组不可哈希
        # 手动管理 reset() 即可
        self._scenes_data: List[SceneData] = []

    def update(
            self,
            point_coords: torch.Tensor,
            gt_instance_ids: torch.Tensor,
            gt_semantic: torch.Tensor,
            pred_instance_ids: torch.Tensor,
            pred_semantic: torch.Tensor,
    ) -> None:
        """更新内部状态，累积一个场景的数据。

        Args:
            point_coords: Tensor, shape (N, 3)
                点坐标。
            gt_instance_ids: LongTensor, shape (N,)
                真值实例 ID（-1 或 num_classes 以上表示 void）。
            gt_semantic: LongTensor, shape (N,)
                真值语义标签。
            pred_instance_ids: LongTensor, shape (N,)
                预测实例 ID。
            pred_semantic: LongTensor, shape (N,)
                预测语义标签。
        """
        # 转为 CPU numpy 以便后续使用 scipy
        scene = SceneData(
            point_coords=point_coords.detach().cpu().numpy().astype(
                np.float32),
            gt_instance_ids=gt_instance_ids.detach().cpu().numpy().astype(
                np.int64),
            gt_semantic=gt_semantic.detach().cpu().numpy().astype(np.int64),
            pred_instance_ids=pred_instance_ids.detach().cpu().numpy(
                ).astype(np.int64),
            pred_semantic=pred_semantic.detach().cpu().numpy().astype(
                np.int64),
        )
        self._scenes_data.append(scene)

    def reset(self) -> None:
        """重置内部状态。"""
        super().reset()
        self._scenes_data = []

    def compute(self) -> BoundaryPanopticMetricResults:
        """在所有累积场景上计算 Boundary Panoptic 指标。

        Returns:
            BoundaryPanopticMetricResults: 包含 bpq, bsq, brq, biou
                及逐类别细分得分。
        """
        num_classes = self.num_classes

        if len(self._scenes_data) == 0:
            log.warning(
                "BoundaryPanopticQuality3D: 无可用预测数据，返回 NaN。")
            return self._empty_results()

        # 汇总所有场景的逐类别统计量
        total_tp = np.zeros(num_classes, dtype=np.float64)
        total_fp = np.zeros(num_classes, dtype=np.float64)
        total_fn = np.zeros(num_classes, dtype=np.float64)
        total_bsq_sum = np.zeros(num_classes, dtype=np.float64)
        total_biou_sum = np.zeros(num_classes, dtype=np.float64)

        for scene in self._scenes_data:
            result = compute_boundary_panoptic_metrics(
                scene.point_coords,
                scene.gt_instance_ids,
                scene.gt_semantic,
                scene.pred_instance_ids,
                scene.pred_semantic,
                num_classes,
                self.stuff_classes,
                self.boundary_distance,
                self.ignore_unseen_classes,
            )
            total_tp += result['tp_per_class']
            total_fp += result['fp_per_class']
            total_fn += result['fn_per_class']
            total_bsq_sum += result['bsq_sum_per_class']
            total_biou_sum += result['biou_sum_per_class']

        # 逐类别指标计算
        bsq_per_class = np.zeros(num_classes, dtype=np.float64)
        brq_per_class = np.zeros(num_classes, dtype=np.float64)
        biou_per_class = np.zeros(num_classes, dtype=np.float64)

        for c in range(num_classes):
            tp = total_tp[c]
            fp = total_fp[c]
            fn = total_fn[c]

            if tp > 0:
                bsq_per_class[c] = total_bsq_sum[c] / tp
                biou_per_class[c] = total_biou_sum[c] / tp
            else:
                bsq_per_class[c] = 0.0
                biou_per_class[c] = 0.0

            denom = tp + 0.5 * fp + 0.5 * fn
            if denom > 0:
                brq_per_class[c] = tp / denom
            else:
                brq_per_class[c] = 0.0

        bpq_per_class = bsq_per_class * brq_per_class

        # 标记可见类别（至少有一个 GT 或 Pred 实例）
        is_seen = (total_tp + total_fp + total_fn) > 0

        # 处理未见类别
        if self.ignore_unseen_classes:
            bpq_per_class[~is_seen] = np.nan
            bsq_per_class[~is_seen] = np.nan
            brq_per_class[~is_seen] = np.nan
            biou_per_class[~is_seen] = np.nan

        # stuff/thing 分组
        is_stuff = np.array(
            [i in self.stuff_classes for i in range(num_classes)], dtype=bool)
        has_stuff = is_stuff.any()

        # 转为 torch tensor
        device = torch.device('cpu')
        bpq_t = torch.from_numpy(bpq_per_class).float().to(device)
        bsq_t = torch.from_numpy(bsq_per_class).float().to(device)
        brq_t = torch.from_numpy(brq_per_class).float().to(device)
        biou_t = torch.from_numpy(biou_per_class).float().to(device)

        # 构建结果
        metrics = BoundaryPanopticMetricResults()
        metrics.bpq = bpq_t.nanmean()
        metrics.bsq = bsq_t.nanmean()
        metrics.brq = brq_t.nanmean()
        metrics.biou = biou_t.nanmean()
        metrics.bpq_thing = bpq_t[~is_stuff].nanmean()
        metrics.bsq_thing = bsq_t[~is_stuff].nanmean()
        metrics.brq_thing = brq_t[~is_stuff].nanmean()
        metrics.bpq_stuff = (
            bpq_t[is_stuff].nanmean() if has_stuff
            else torch.tensor(float('nan')))
        metrics.bsq_stuff = (
            bsq_t[is_stuff].nanmean() if has_stuff
            else torch.tensor(float('nan')))
        metrics.brq_stuff = (
            brq_t[is_stuff].nanmean() if has_stuff
            else torch.tensor(float('nan')))
        metrics.bpq_per_class = bpq_t
        metrics.bsq_per_class = bsq_t
        metrics.brq_per_class = brq_t
        metrics.biou_per_class = biou_t
        metrics.tp_per_class = torch.from_numpy(total_tp).float().to(device)
        metrics.fp_per_class = torch.from_numpy(total_fp).float().to(device)
        metrics.fn_per_class = torch.from_numpy(total_fn).float().to(device)

        return metrics

    def _empty_results(self) -> BoundaryPanopticMetricResults:
        """返回空结果（全 NaN），用于无数据时的安全退出。

        Returns:
            BoundaryPanopticMetricResults: 全 NaN 的结果对象。
        """
        device = torch.device('cpu')
        nan_scalar = torch.tensor(float('nan'), device=device)
        nan_vector = torch.full(
            (self.num_classes,), float('nan'), device=device)
        zero_vector = torch.zeros(self.num_classes, device=device)

        metrics = BoundaryPanopticMetricResults()
        for k in ['bpq', 'bsq', 'brq', 'biou',
                   'bpq_thing', 'bsq_thing', 'brq_thing',
                   'bpq_stuff', 'bsq_stuff', 'brq_stuff']:
            setattr(metrics, k, nan_scalar)
        for k in ['bpq_per_class', 'bsq_per_class', 'brq_per_class',
                   'biou_per_class']:
            setattr(metrics, k, nan_vector)
        for k in ['tp_per_class', 'fp_per_class', 'fn_per_class']:
            setattr(metrics, k, zero_vector)

        return metrics
