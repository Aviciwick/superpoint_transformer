from typing import Dict, Iterable, List, Sequence

import numpy as np

from src.manuscript_tools.semantic_boundary import extract_semantic_boundary


__all__ = [
    "build_mixed_superpoint_scene_records",
    "aggregate_mixed_superpoint_records",
]


def _as_numpy(array) -> np.ndarray:
    if isinstance(array, np.ndarray):
        return array
    if hasattr(array, "detach"):
        return array.detach().cpu().numpy()
    return np.asarray(array)


def _ensure_dense_metric(values, num_superpoints: int) -> np.ndarray:
    if values is None:
        return np.zeros(num_superpoints, dtype=np.float32)
    values = _as_numpy(values)
    if values.shape[0] != num_superpoints:
        raise ValueError(
            f"Expected dense superpoint metric with length {num_superpoints}, got shape {values.shape}"
        )
    return values.astype(np.float32, copy=False)


def _bucket_summary(
    records: Dict[str, np.ndarray],
    mask: np.ndarray,
    label: str,
    total_spt_errors: float,
) -> Dict[str, float]:
    count = int(mask.sum())
    if count == 0:
        return {
            "bucket": label,
            "superpoints": 0,
            "points": 0,
            "candidate_hit_rate": 0.0,
            "spt_error_rate": 0.0,
            "spt_error_share": 0.0,
            "bsr_point_acc_gain": 0.0,
            "secondary_slot_mass": 0.0,
            "dual_slot_activation": 0.0,
            "assignment_entropy": 0.0,
            "slot_diversity": 0.0,
            "mean_purity": 0.0,
            "mean_mixing_ratio": 0.0,
            "mean_boundary_fraction": 0.0,
        }

    point_count = float(records["point_count"][mask].sum())
    candidate_mask = mask & records["candidate_mask"]
    candidate_count = int(candidate_mask.sum())
    spt_errors = float(records["spt_errors"][mask].sum())
    bsr_errors = float(records["bsr_errors"][mask].sum())

    return {
        "bucket": label,
        "superpoints": count,
        "points": int(point_count),
        "candidate_hit_rate": float(records["candidate_mask"][mask].mean()),
        "spt_error_rate": (spt_errors / point_count) if point_count > 0 else 0.0,
        "spt_error_share": (spt_errors / total_spt_errors) if total_spt_errors > 0 else 0.0,
        "bsr_point_acc_gain": ((spt_errors - bsr_errors) / point_count) if point_count > 0 else 0.0,
        "secondary_slot_mass": (
            float(records["secondary_slot_mass"][candidate_mask].mean())
            if candidate_count > 0
            else 0.0
        ),
        "dual_slot_activation": (
            float(records["dual_slot_active"][candidate_mask].mean())
            if candidate_count > 0
            else 0.0
        ),
        "assignment_entropy": (
            float(records["assignment_entropy"][candidate_mask].mean())
            if candidate_count > 0
            else 0.0
        ),
        "slot_diversity": (
            float(records["slot_diversity"][candidate_mask].mean())
            if candidate_count > 0
            else 0.0
        ),
        "mean_purity": float(records["purity"][mask].mean()),
        "mean_mixing_ratio": float(records["mixing_ratio"][mask].mean()),
        "mean_boundary_fraction": float(records["boundary_fraction"][mask].mean()),
    }


def build_mixed_superpoint_scene_records(
    point_coords,
    gt_labels,
    baseline_pred,
    bsr_pred,
    super_index,
    num_classes: int,
    boundary_distance: float = None,
    boundary_band_k: int = None,
    candidate_mask=None,
    assignment_entropy=None,
    secondary_slot_mass=None,
    slot_diversity=None,
    dual_slot_threshold: float = 0.2,
) -> Dict[str, np.ndarray]:
    point_coords = _as_numpy(point_coords).astype(np.float32, copy=False)
    gt_labels = _as_numpy(gt_labels).astype(np.int64, copy=False)
    baseline_pred = _as_numpy(baseline_pred).astype(np.int64, copy=False)
    bsr_pred = _as_numpy(bsr_pred).astype(np.int64, copy=False)
    super_index = _as_numpy(super_index).astype(np.int64, copy=False)

    if super_index.ndim != 1:
        raise ValueError(f"Expected 1D super_index, got shape {super_index.shape}")
    if gt_labels.shape[0] != super_index.shape[0]:
        raise ValueError("gt_labels and super_index must have the same length")

    num_superpoints = int(super_index.max()) + 1 if super_index.size > 0 else 0
    dense_candidate_mask = _ensure_dense_metric(candidate_mask, num_superpoints).astype(bool)
    dense_assignment_entropy = _ensure_dense_metric(assignment_entropy, num_superpoints)
    dense_secondary_slot_mass = _ensure_dense_metric(secondary_slot_mass, num_superpoints)
    dense_slot_diversity = _ensure_dense_metric(slot_diversity, num_superpoints)

    valid = (gt_labels >= 0) & (gt_labels < num_classes)
    point_count = np.bincount(super_index[valid], minlength=num_superpoints).astype(np.int64)
    class_hist = np.zeros((num_superpoints, num_classes), dtype=np.int64)
    np.add.at(class_hist, (super_index[valid], gt_labels[valid]), 1)

    dominant_counts = class_hist.max(axis=1) if num_superpoints > 0 else np.zeros(0, dtype=np.int64)
    purity = np.divide(
        dominant_counts,
        np.maximum(point_count, 1),
        out=np.zeros(num_superpoints, dtype=np.float32),
        where=point_count > 0,
    )
    mixing_ratio = 1.0 - purity

    boundary_mask = extract_semantic_boundary(
        point_coords=point_coords,
        semantic_labels=gt_labels,
        boundary_distance=boundary_distance,
        boundary_band_k=boundary_band_k,
        num_classes=num_classes,
    )
    boundary_count = np.bincount(super_index[boundary_mask & valid], minlength=num_superpoints).astype(np.int64)
    boundary_fraction = np.divide(
        boundary_count,
        np.maximum(point_count, 1),
        out=np.zeros(num_superpoints, dtype=np.float32),
        where=point_count > 0,
    )

    spt_errors = np.bincount(
        super_index[valid & (baseline_pred != gt_labels)],
        minlength=num_superpoints,
    ).astype(np.int64)
    bsr_errors = np.bincount(
        super_index[valid & (bsr_pred != gt_labels)],
        minlength=num_superpoints,
    ).astype(np.int64)

    return {
        "point_count": point_count,
        "purity": purity.astype(np.float32, copy=False),
        "mixing_ratio": mixing_ratio.astype(np.float32, copy=False),
        "boundary_fraction": boundary_fraction.astype(np.float32, copy=False),
        "candidate_mask": dense_candidate_mask,
        "assignment_entropy": dense_assignment_entropy,
        "secondary_slot_mass": dense_secondary_slot_mass,
        "slot_diversity": dense_slot_diversity,
        "dual_slot_active": dense_secondary_slot_mass > float(dual_slot_threshold),
        "spt_errors": spt_errors,
        "bsr_errors": bsr_errors,
    }


def aggregate_mixed_superpoint_records(
    scene_records: Iterable[Dict[str, np.ndarray]],
    purity_thresholds: Sequence[float] = (0.95, 0.8),
    transition_thresholds: Sequence[float] = (0.05, 0.2),
) -> Dict[str, object]:
    records_list: List[Dict[str, np.ndarray]] = list(scene_records)
    if not records_list:
        return {
            "num_scenes": 0,
            "num_superpoints": 0,
            "purity_buckets": [],
            "transition_buckets": [],
            "candidate_prior": {},
        }

    merged = {
        key: np.concatenate([scene[key] for scene in records_list], axis=0)
        for key in records_list[0].keys()
    }
    total_spt_errors = float(merged["spt_errors"].sum())
    purity_clean, purity_mixed = float(purity_thresholds[0]), float(purity_thresholds[1])
    transition_low, transition_high = float(transition_thresholds[0]), float(transition_thresholds[1])

    purity_masks = [
        ("clean", merged["purity"] >= purity_clean),
        (
            "moderately_mixed",
            (merged["purity"] < purity_clean) & (merged["purity"] >= purity_mixed),
        ),
        ("heavily_mixed", merged["purity"] < purity_mixed),
    ]
    transition_masks = [
        ("non_transition", merged["boundary_fraction"] < transition_low),
        (
            "transition",
            (merged["boundary_fraction"] >= transition_low)
            & (merged["boundary_fraction"] < transition_high),
        ),
        ("transition_heavy", merged["boundary_fraction"] >= transition_high),
    ]
    mixed_content_mask = merged["purity"] < purity_mixed
    transition_heavy_exclusive_mask = (~mixed_content_mask) & (
        merged["boundary_fraction"] >= transition_high
    )
    clean_exclusive_mask = (
        (~mixed_content_mask)
        & (merged["boundary_fraction"] < transition_high)
        & (merged["purity"] >= purity_clean)
    )
    diagnostic_slice_masks = [
        ("clean", clean_exclusive_mask),
        ("transition_heavy", transition_heavy_exclusive_mask),
        ("mixed_content", mixed_content_mask),
    ]
    candidate_mask = merged["candidate_mask"].astype(bool)
    mixed_mask = merged["purity"] < purity_mixed
    transition_mask = merged["boundary_fraction"] >= transition_low
    transition_heavy_mask = merged["boundary_fraction"] >= transition_high
    candidate_count = int(candidate_mask.sum())

    candidate_prior = {
        "candidate_rate": float(candidate_mask.mean()) if candidate_mask.size > 0 else 0.0,
        "transition_hit_rate": (
            float(candidate_mask[transition_heavy_mask].mean())
            if transition_heavy_mask.any()
            else 0.0
        ),
        "mixed_superpoint_coverage": (
            float(candidate_mask[mixed_mask].mean())
            if mixed_mask.any()
            else 0.0
        ),
        "candidate_boundary_precision": (
            float(transition_mask[candidate_mask].mean())
            if candidate_count > 0
            else 0.0
        ),
        "candidate_transition_heavy_precision": (
            float(transition_heavy_mask[candidate_mask].mean())
            if candidate_count > 0
            else 0.0
        ),
        "candidate_mixed_precision": (
            float(mixed_mask[candidate_mask].mean())
            if candidate_count > 0
            else 0.0
        ),
    }

    return {
        "num_scenes": len(records_list),
        "num_superpoints": int(merged["point_count"].shape[0]),
        "total_points": int(merged["point_count"].sum()),
        "purity_thresholds": {
            "clean": purity_clean,
            "moderately_mixed": purity_mixed,
        },
        "transition_thresholds": {
            "transition": transition_low,
            "transition_heavy": transition_high,
        },
        "diagnostic_slice_rules": {
            "mutually_exclusive": True,
            "assignment_order": ["mixed_content", "transition_heavy", "clean"],
            "clean": f"purity >= {purity_clean} after mixed/transition assignment",
            "transition_heavy": (
                f"boundary_fraction >= {transition_high} after mixed assignment"
            ),
            "mixed_content": f"purity < {purity_mixed}",
            "excluded": (
                f"{purity_mixed} <= purity < {purity_clean} and "
                f"boundary_fraction < {transition_high}"
            ),
        },
        "diagnostic_slices": [
            _bucket_summary(merged, mask, label, total_spt_errors)
            for label, mask in diagnostic_slice_masks
        ],
        "purity_buckets": [
            _bucket_summary(merged, mask, label, total_spt_errors)
            for label, mask in purity_masks
        ],
        "transition_buckets": [
            _bucket_summary(merged, mask, label, total_spt_errors)
            for label, mask in transition_masks
        ],
        "candidate_prior": candidate_prior,
    }
