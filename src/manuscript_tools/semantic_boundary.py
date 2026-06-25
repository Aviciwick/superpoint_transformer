from typing import Dict, Iterable, List

import numpy as np

__all__ = [
    "extract_3d_boundary",
    "extract_semantic_boundary",
    "extract_semantic_boundary_knn",
    "compute_semantic_boundary_scene_stats",
    "aggregate_semantic_boundary_stats",
]


def _as_numpy(array) -> np.ndarray:
    if isinstance(array, np.ndarray):
        return array
    if hasattr(array, "detach"):
        return array.detach().cpu().numpy()
    return np.asarray(array)


def _confusion_matrix(
    pred_labels: np.ndarray,
    gt_labels: np.ndarray,
    num_classes: int = None,
) -> np.ndarray:
    if num_classes is None:
        raise ValueError("num_classes must be provided")
    confusion = np.zeros((num_classes, num_classes), dtype=np.int64)
    valid = (gt_labels >= 0) & (gt_labels < num_classes)
    if not valid.any():
        return confusion

    pred = pred_labels[valid].astype(np.int64, copy=False)
    target = gt_labels[valid].astype(np.int64, copy=False)
    np.add.at(confusion, (pred, target), 1)
    return confusion


def _miou_from_confusion(confusion: np.ndarray) -> float:
    tp = np.diag(confusion).astype(np.float64)
    pred_sum = confusion.sum(axis=1).astype(np.float64)
    gt_sum = confusion.sum(axis=0).astype(np.float64)
    union = pred_sum + gt_sum - tp
    valid = union > 0
    if not valid.any():
        return 0.0
    iou = tp[valid] / np.maximum(union[valid], 1e-8)
    return float(iou.mean() * 100.0)


def extract_3d_boundary(point_coords, mask, d):
    from scipy.spatial import cKDTree

    boundary = np.zeros(mask.shape[0], dtype=bool)

    inside_idx = np.where(mask)[0]
    outside_idx = np.where(~mask)[0]
    if inside_idx.size == 0 or outside_idx.size == 0:
        return boundary

    inside_coords = point_coords[inside_idx]
    bbox_min = inside_coords.min(axis=0) - d
    bbox_max = inside_coords.max(axis=0) + d

    outside_coords = point_coords[outside_idx]
    local_mask = np.all(
        (outside_coords >= bbox_min) & (outside_coords <= bbox_max), axis=1)
    local_outside_coords = outside_coords[local_mask]

    if local_outside_coords.shape[0] == 0:
        return boundary

    tree = cKDTree(local_outside_coords)
    dists, _ = tree.query(inside_coords, k=1)
    is_boundary = dists < d
    boundary[inside_idx[is_boundary]] = True
    return boundary


def extract_semantic_boundary(
    point_coords,
    semantic_labels,
    boundary_distance: float = None,
    num_classes: int = None,
    boundary_band_k: int = None,
) -> np.ndarray:
    if num_classes is None:
        raise ValueError("num_classes must be provided")
    if boundary_band_k is not None:
        return extract_semantic_boundary_knn(
            point_coords=point_coords,
            semantic_labels=semantic_labels,
            boundary_band_k=boundary_band_k,
            num_classes=num_classes,
        )
    if boundary_distance is None:
        boundary_distance = 0.5

    point_coords = _as_numpy(point_coords).astype(np.float32, copy=False)
    semantic_labels = _as_numpy(semantic_labels).astype(np.int64, copy=False)

    boundary_mask = np.zeros(semantic_labels.shape[0], dtype=bool)
    valid = (semantic_labels >= 0) & (semantic_labels < num_classes)
    if not valid.any():
        return boundary_mask

    valid_coords = point_coords[valid]
    valid_labels = semantic_labels[valid]
    valid_boundary = np.zeros(valid_labels.shape[0], dtype=bool)

    for class_id in np.unique(valid_labels):
        class_mask = valid_labels == class_id
        if class_mask.any() and (~class_mask).any():
            valid_boundary |= extract_3d_boundary(valid_coords, class_mask, float(boundary_distance))

    boundary_mask[valid] = valid_boundary
    return boundary_mask


def extract_semantic_boundary_knn(
    point_coords,
    semantic_labels,
    boundary_band_k: int,
    num_classes: int,
) -> np.ndarray:
    from scipy.spatial import cKDTree

    point_coords = _as_numpy(point_coords).astype(np.float32, copy=False)
    semantic_labels = _as_numpy(semantic_labels).astype(np.int64, copy=False)

    boundary_mask = np.zeros(semantic_labels.shape[0], dtype=bool)
    valid = (semantic_labels >= 0) & (semantic_labels < num_classes)
    valid_count = int(valid.sum())
    if valid_count <= 1:
        return boundary_mask

    valid_coords = point_coords[valid]
    valid_labels = semantic_labels[valid]
    k_query = min(int(boundary_band_k) + 1, valid_count)
    if k_query <= 1:
        return boundary_mask

    tree = cKDTree(valid_coords)
    _, neighbors = tree.query(valid_coords, k=k_query)
    if neighbors.ndim == 1:
        neighbors = neighbors[:, None]
    neighbor_labels = valid_labels[neighbors[:, 1:]]
    valid_boundary = (neighbor_labels != valid_labels[:, None]).any(axis=1)
    boundary_mask[valid] = valid_boundary
    return boundary_mask


def compute_semantic_boundary_scene_stats(
    point_coords,
    gt_semantic,
    pred_semantic,
    num_classes: int,
    boundary_distance: float = None,
    boundary_band_k: int = None,
) -> Dict[str, np.ndarray]:
    point_coords = _as_numpy(point_coords).astype(np.float32, copy=False)
    gt_semantic = _as_numpy(gt_semantic).astype(np.int64, copy=False)
    pred_semantic = _as_numpy(pred_semantic).astype(np.int64, copy=False)

    valid = (gt_semantic >= 0) & (gt_semantic < num_classes)
    if boundary_band_k is None and boundary_distance is None:
        boundary_band_k = 16
    gt_boundary = extract_semantic_boundary(
        point_coords=point_coords,
        semantic_labels=gt_semantic,
        boundary_distance=boundary_distance,
        boundary_band_k=boundary_band_k,
        num_classes=num_classes,
    )
    pred_boundary = extract_semantic_boundary(
        point_coords=point_coords,
        semantic_labels=pred_semantic,
        boundary_distance=boundary_distance,
        boundary_band_k=boundary_band_k,
        num_classes=num_classes,
    )

    boundary_valid = valid & gt_boundary
    boundary_intersection = int(np.logical_and(gt_boundary, pred_boundary & valid).sum())
    boundary_union = int(np.logical_or(gt_boundary, pred_boundary & valid).sum())
    pred_boundary_valid = pred_boundary & valid

    return {
        "overall_confusion": _confusion_matrix(pred_semantic[valid], gt_semantic[valid], num_classes),
        "transition_confusion": _confusion_matrix(
            pred_semantic[boundary_valid],
            gt_semantic[boundary_valid],
            num_classes,
        ),
        "boundary_intersection": np.asarray(boundary_intersection, dtype=np.int64),
        "boundary_union": np.asarray(boundary_union, dtype=np.int64),
        "boundary_gt_points": np.asarray(int(gt_boundary.sum()), dtype=np.int64),
        "boundary_pred_points": np.asarray(int(pred_boundary_valid.sum()), dtype=np.int64),
        "valid_points": np.asarray(int(valid.sum()), dtype=np.int64),
    }


def aggregate_semantic_boundary_stats(scene_stats: Iterable[Dict[str, np.ndarray]]) -> Dict[str, float]:
    stats_list: List[Dict[str, np.ndarray]] = list(scene_stats)
    if not stats_list:
        return {
            "overall_miou": 0.0,
            "transition_miou": 0.0,
            "transition_region_miou": 0.0,
            "boundary_iou": 0.0,
            "boundary_f1": 0.0,
            "boundary_precision": 0.0,
            "boundary_recall": 0.0,
            "boundary_gt_fraction": 0.0,
            "num_scenes": 0,
            "num_valid_points": 0,
        }

    overall_confusion = np.sum([s["overall_confusion"] for s in stats_list], axis=0)
    transition_confusion = np.sum([s["transition_confusion"] for s in stats_list], axis=0)
    boundary_intersection = int(np.sum([s["boundary_intersection"] for s in stats_list]))
    boundary_union = int(np.sum([s["boundary_union"] for s in stats_list]))
    boundary_gt_points = int(np.sum([s["boundary_gt_points"] for s in stats_list]))
    boundary_pred_points = int(np.sum([s["boundary_pred_points"] for s in stats_list]))
    valid_points = int(np.sum([s["valid_points"] for s in stats_list]))

    precision = (
        float(boundary_intersection) / float(boundary_pred_points)
        if boundary_pred_points > 0
        else 0.0
    )
    recall = (
        float(boundary_intersection) / float(boundary_gt_points)
        if boundary_gt_points > 0
        else 0.0
    )
    boundary_f1 = (
        2.0 * precision * recall / max(precision + recall, 1e-8)
        if precision > 0.0 or recall > 0.0
        else 0.0
    )
    transition_miou = _miou_from_confusion(transition_confusion)

    return {
        "overall_miou": _miou_from_confusion(overall_confusion),
        "transition_miou": transition_miou,
        "transition_region_miou": transition_miou,
        "boundary_iou": float(boundary_intersection) / float(boundary_union) if boundary_union > 0 else 1.0,
        "boundary_f1": boundary_f1,
        "boundary_precision": precision,
        "boundary_recall": recall,
        "boundary_gt_fraction": float(boundary_gt_points) / float(valid_points) if valid_points > 0 else 0.0,
        "num_scenes": len(stats_list),
        "num_valid_points": valid_points,
    }
