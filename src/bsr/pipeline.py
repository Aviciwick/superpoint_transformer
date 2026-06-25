from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .partition_adapter import OptionalPartitionAdapter
from .refiner import PointSuperpointMixtureRefiner
from .selector import BoundaryPriorSelector


__all__ = [
    "BSROutput",
    "BSRModule",
    "build_candidate_point_cloud",
    "build_packed_points",
    "run_bsr",
    "compute_bsr_losses",
]


@dataclass
class BSROutput:
    candidate_indices: torch.Tensor
    candidate_scores: torch.Tensor
    refined_sp_features: torch.Tensor
    refined_sp_logits: torch.Tensor
    sampled_point_logits: torch.Tensor
    point_logits: Optional[torch.Tensor] = None
    point_propagation_mask: Optional[torch.Tensor] = None
    point_propagation_coverage: float = 0.0
    slot_affinity: Optional[torch.Tensor] = None
    coarse_candidate_logits: Optional[torch.Tensor] = None
    fused_candidate_logits: Optional[torch.Tensor] = None
    sampled_point_indices: Optional[torch.Tensor] = None
    sampled_point_mask: Optional[torch.Tensor] = None
    boundary_logits: Optional[torch.Tensor] = None
    boundary_scores: Optional[torch.Tensor] = None
    point_residual_gates: Optional[torch.Tensor] = None
    slot_logits: Optional[torch.Tensor] = None
    slot_assignments: Optional[torch.Tensor] = None
    slot_masses: Optional[torch.Tensor] = None
    slot_tokens: Optional[torch.Tensor] = None
    assignment_entropy: Optional[torch.Tensor] = None
    secondary_slot_mass: Optional[torch.Tensor] = None
    dual_slot_activation_ratio: float = 0.0
    slot_diversity: Optional[torch.Tensor] = None
    selector_score_terms: Optional[Dict[str, torch.Tensor]] = None
    selector_score_summary: Optional[Dict[str, Dict[str, float]]] = None
    num_superpoints: int = 0
    num_candidates: int = 0
    candidate_ratio: float = 0.0
    num_valid_sampled_points: int = 0
    avg_valid_points_per_candidate: float = 0.0
    effective_refine_ratio: float = 0.0
    sampling_mode: Optional[str] = None
    point_head_mode: Optional[str] = None
    refiner_variant: Optional[str] = None
    warmup_loss_weight: float = 1.0
    warmup_fusion_alpha: float = 0.0
    partition_output: Optional[object] = None


def _normalize_vector(values: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    if values.numel() == 0:
        return values
    min_value = values.min()
    max_value = values.max()
    return (values - min_value) / (max_value - min_value + eps)


def _repeat_to_length(indices: torch.Tensor, target_length: int) -> torch.Tensor:
    if indices.numel() == 0:
        return indices
    if indices.numel() >= target_length:
        return indices[:target_length]
    repeats = (target_length + indices.numel() - 1) // indices.numel()
    return indices.repeat(repeats)[:target_length]


def _sample_evenly(sorted_indices: torch.Tensor, num_samples: int) -> torch.Tensor:
    if sorted_indices.numel() <= num_samples:
        return sorted_indices
    positions = torch.linspace(
        0,
        sorted_indices.numel() - 1,
        steps=num_samples,
        device=sorted_indices.device,
    ).round().long()
    return sorted_indices[positions]


def _summarize_selector_terms(
    selector_score_terms: Optional[Dict[str, torch.Tensor]],
) -> Dict[str, Dict[str, float]]:
    summary: Dict[str, Dict[str, float]] = {}
    if not selector_score_terms:
        return summary

    for key, values in selector_score_terms.items():
        if values is None or values.numel() == 0:
            summary[key] = {"mean": 0.0, "var": 0.0}
            continue
        values_f = values.detach().to(dtype=torch.float32)
        summary[key] = {
            "mean": float(values_f.mean().item()),
            "var": float(values_f.var(unbiased=False).item()),
        }
    return summary


def _compute_output_diagnostics(
    num_superpoints: int,
    num_candidates: int,
    sampled_point_mask: Optional[torch.Tensor],
    sampled_point_logits: Optional[torch.Tensor],
) -> Tuple[float, int, float, float]:
    if num_superpoints <= 0:
        return 0.0, 0, 0.0, 0.0

    if sampled_point_mask is not None:
        num_valid_sampled_points = int(sampled_point_mask.detach().sum().item())
        points_per_candidate = int(sampled_point_mask.shape[1]) if sampled_point_mask.dim() == 2 else 0
    elif sampled_point_logits is not None and sampled_point_logits.dim() >= 2:
        num_valid_sampled_points = int(sampled_point_logits.shape[0] * sampled_point_logits.shape[1])
        points_per_candidate = int(sampled_point_logits.shape[1])
    else:
        num_valid_sampled_points = 0
        points_per_candidate = 0

    candidate_ratio = float(num_candidates) / float(num_superpoints)
    avg_valid_points_per_candidate = (
        float(num_valid_sampled_points) / float(num_candidates) if num_candidates > 0 else 0.0
    )
    total_candidate_budget = max(num_superpoints * max(points_per_candidate, 1), 1)
    effective_refine_ratio = float(num_valid_sampled_points) / float(total_candidate_budget)
    return (
        candidate_ratio,
        num_valid_sampled_points,
        avg_valid_points_per_candidate,
        effective_refine_ratio,
    )


def _select_point_samples(
    point_positions: torch.Tensor,
    point_features: torch.Tensor,
    centroid: torch.Tensor,
    n_sample: int,
    sampling_mode: str,
    sampling_without_replacement: bool,
) -> Tuple[torch.Tensor, torch.Tensor]:
    num_points = point_positions.shape[0]
    device = point_positions.device
    sampled_indices = torch.full((n_sample,), -1, dtype=torch.long, device=device)
    sampled_mask = torch.zeros(n_sample, dtype=torch.bool, device=device)

    if num_points == 0 or n_sample <= 0:
        return sampled_indices, sampled_mask

    if sampling_mode not in {"random", "coverage", "boundary_hybrid"}:
        raise ValueError(
            f"Unsupported BSR point sampling mode '{sampling_mode}'. "
            "Expected one of {'random', 'coverage', 'boundary_hybrid'}."
        )

    if sampling_mode == "random":
        if sampling_without_replacement:
            base_indices = torch.randperm(num_points, device=device)[: min(num_points, n_sample)]
        else:
            base_indices = torch.randint(0, num_points, (n_sample,), device=device)
    else:
        radial_distance = torch.norm(point_positions - centroid.unsqueeze(0), dim=1)
        radial_order = torch.argsort(radial_distance)
        budget = min(num_points, n_sample)

        if sampling_mode == "coverage":
            base_indices = _sample_evenly(radial_order, budget)
        else:
            boundary_quota = min(budget, max(1, (budget + 1) // 2))
            coverage_quota = max(0, budget - boundary_quota)
            radial_score = _normalize_vector(radial_distance)
            if point_features.shape[1] > 3:
                attr_dev = torch.norm(
                    point_features[:, 3:] - point_features[:, 3:].mean(dim=0, keepdim=True),
                    dim=1,
                )
                attr_score = _normalize_vector(attr_dev)
            else:
                attr_score = torch.zeros_like(radial_score)

            boundary_score = radial_score + 0.5 * attr_score
            boundary_order = torch.argsort(boundary_score, descending=True)
            boundary_pick = boundary_order[:boundary_quota]

            selected_mask = torch.zeros(num_points, dtype=torch.bool, device=device)
            selected_mask[boundary_pick] = True
            coverage_pick = boundary_pick.new_empty(0)
            if coverage_quota > 0:
                remaining = torch.arange(num_points, device=device)[~selected_mask]
                if remaining.numel() > 0:
                    remaining_order = remaining[torch.argsort(radial_distance[remaining])]
                    coverage_pick = _sample_evenly(
                        remaining_order,
                        min(coverage_quota, remaining_order.numel()),
                    )
                    selected_mask[coverage_pick] = True

            base_indices = torch.cat([boundary_pick, coverage_pick], dim=0)
            if base_indices.numel() < budget:
                extras = boundary_order[~selected_mask[boundary_order]][: budget - base_indices.numel()]
                base_indices = torch.cat([base_indices, extras], dim=0)

        if not sampling_without_replacement and base_indices.numel() < n_sample:
            base_indices = _repeat_to_length(base_indices, n_sample)

    if sampling_without_replacement:
        sampled_indices[: base_indices.numel()] = base_indices
        sampled_mask[: base_indices.numel()] = True
    else:
        sampled_indices.copy_(base_indices[:n_sample])
        sampled_mask.fill_(True)

    return sampled_indices, sampled_mask


def _build_random_packed_indices(
    order: torch.Tensor,
    super_index: torch.Tensor,
    counts: torch.Tensor,
    ptr: torch.Tensor,
    superpoint_indices: torch.Tensor,
    n_sample: int,
    sampling_without_replacement: bool,
) -> Tuple[torch.Tensor, torch.Tensor]:
    device = super_index.device
    num_superpoints = int(superpoint_indices.numel())
    packed_point_idx = torch.full((num_superpoints, n_sample), -1, dtype=torch.long, device=device)
    packed_mask = torch.zeros((num_superpoints, n_sample), dtype=torch.bool, device=device)
    active_counts = counts[superpoint_indices]

    if num_superpoints == 0:
        return packed_point_idx, packed_mask

    if not sampling_without_replacement:
        counts_clamped = active_counts.clamp(min=1)
        random_offsets = torch.randint(
            0,
            2**31 - 1,
            (num_superpoints, n_sample),
            device=device,
            dtype=torch.long,
        ) % counts_clamped.unsqueeze(1)
        global_idx = ptr[superpoint_indices].unsqueeze(1) + random_offsets
        packed_point_idx = order[global_idx]
        packed_mask = (active_counts > 0).unsqueeze(1).expand(-1, n_sample)
        packed_point_idx[~packed_mask] = -1
        return packed_point_idx, packed_mask

    row_map = torch.full((counts.shape[0],), -1, dtype=torch.long, device=device)
    row_map[superpoint_indices] = torch.arange(num_superpoints, device=device)

    shuffled = torch.randperm(super_index.numel(), device=device)
    shuffled_super = super_index[shuffled]
    grouped_order = shuffled[torch.argsort(shuffled_super, stable=True)]
    grouped_super = super_index[grouped_order]
    local_rank = torch.arange(grouped_order.numel(), device=device) - ptr[grouped_super]
    keep = local_rank < n_sample

    chosen_idx = grouped_order[keep]
    chosen_super = grouped_super[keep]
    chosen_rank = local_rank[keep]
    chosen_row = row_map[chosen_super]
    chosen_valid = chosen_row >= 0

    packed_point_idx[chosen_row[chosen_valid], chosen_rank[chosen_valid]] = chosen_idx[chosen_valid]
    packed_mask[chosen_row[chosen_valid], chosen_rank[chosen_valid]] = True
    return packed_point_idx, packed_mask


def _build_coverage_packed_indices(
    point_positions: torch.Tensor,
    sp_centroids: Optional[torch.Tensor],
    super_index: torch.Tensor,
    counts: torch.Tensor,
    ptr: torch.Tensor,
    superpoint_indices: torch.Tensor,
    n_sample: int,
    sampling_without_replacement: bool,
) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
    if sp_centroids is None:
        return None

    device = super_index.device
    num_superpoints = int(superpoint_indices.numel())
    packed_point_idx = torch.full((num_superpoints, n_sample), -1, dtype=torch.long, device=device)
    packed_mask = torch.zeros((num_superpoints, n_sample), dtype=torch.bool, device=device)
    if num_superpoints == 0:
        return packed_point_idx, packed_mask

    centroid_per_point = sp_centroids[super_index]
    radial_distance = torch.norm(point_positions - centroid_per_point, dim=1)
    radial_order = torch.argsort(radial_distance, stable=True)
    grouped_super = super_index[radial_order]
    grouped_order = radial_order[torch.argsort(grouped_super, stable=True)]

    active_counts = counts[superpoint_indices]
    step_idx = torch.arange(n_sample, device=device).unsqueeze(0).expand(num_superpoints, -1)

    if sampling_without_replacement:
        sample_counts = active_counts.clamp(max=n_sample)
        packed_mask = step_idx < sample_counts.unsqueeze(1)
        source_span = (active_counts - 1).clamp(min=0).unsqueeze(1).to(torch.float32)
        target_span = (sample_counts - 1).clamp(min=1).unsqueeze(1).to(torch.float32)
        positions = torch.round(step_idx.to(torch.float32) * source_span / target_span).long()
        positions = torch.where(sample_counts.unsqueeze(1) > 1, positions, torch.zeros_like(positions))
    else:
        packed_mask = (active_counts > 0).unsqueeze(1).expand(-1, n_sample)
        source_span = (active_counts - 1).clamp(min=0).unsqueeze(1).to(torch.float32)
        target_span = torch.tensor(max(n_sample - 1, 1), device=device, dtype=torch.float32)
        positions = torch.round(step_idx.to(torch.float32) * source_span / target_span).long()

    max_grouped_idx = max(grouped_order.numel() - 1, 0)
    global_idx = ptr[superpoint_indices].unsqueeze(1) + positions
    safe_idx = global_idx.clamp(min=0, max=max_grouped_idx)
    packed_point_idx[packed_mask] = grouped_order[safe_idx[packed_mask]]
    packed_point_idx[~packed_mask] = -1
    return packed_point_idx, packed_mask


def build_packed_points(
    nag,
    n_sample: int = 64,
    raw_keys: Optional[List[str]] = None,
    superpoint_indices: Optional[torch.Tensor] = None,
    device: Optional[torch.device] = None,
    sampling_mode: str = "coverage",
    sampling_without_replacement: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if raw_keys is None:
        raw_keys = ["pos"]
    if device is None:
        device = nag[0].pos.device

    num_superpoints_total = nag[1].num_nodes if hasattr(nag[1], "num_nodes") else nag[1].pos.shape[0]
    if superpoint_indices is None:
        superpoint_indices = torch.arange(num_superpoints_total, device=device, dtype=torch.long)
    else:
        superpoint_indices = superpoint_indices.to(device=device, dtype=torch.long)

    num_superpoints = int(superpoint_indices.numel())
    super_index = getattr(nag[0], "super_index", None)
    if super_index is None:
        raise ValueError("NAG level-0 data must provide super_index for BSR packing")
    super_index = super_index.to(device)

    counts = torch.bincount(super_index, minlength=num_superpoints_total).to(device)
    order = torch.argsort(super_index)
    ptr = torch.cat([torch.zeros(1, dtype=torch.long, device=device), counts.cumsum(0)])

    raw_tensors = []
    for key in raw_keys:
        if hasattr(nag[0], key):
            attr = getattr(nag[0], key)
            if attr is not None:
                if attr.dim() == 1:
                    attr = attr.unsqueeze(-1)
                raw_tensors.append(attr)
    if not raw_tensors:
        raise ValueError(f"Could not gather raw keys for BSR packing: {raw_keys}")

    raw_data = torch.cat(raw_tensors, dim=-1) if len(raw_tensors) > 1 else raw_tensors[0]
    raw_data = raw_data.to(device, non_blocking=True)

    if num_superpoints == 0:
        d_raw = int(raw_data.shape[-1])
        return (
            torch.zeros(0, n_sample, d_raw, device=device),
            torch.full((0, n_sample), -1, dtype=torch.long, device=device),
            torch.zeros(0, n_sample, dtype=torch.bool, device=device),
        )

    if not hasattr(nag[0], "pos") or nag[0].pos is None:
        raise ValueError("BSR point packing requires level-0 positions stored in nag[0].pos")

    point_positions = nag[0].pos.to(device, non_blocking=True)
    sp_centroids = getattr(nag[1], "pos", None)
    if sp_centroids is not None:
        sp_centroids = sp_centroids.to(device, non_blocking=True)

    if sampling_mode == "random":
        packed_point_idx, packed_mask = _build_random_packed_indices(
            order=order,
            super_index=super_index,
            counts=counts,
            ptr=ptr,
            superpoint_indices=superpoint_indices,
            n_sample=n_sample,
            sampling_without_replacement=sampling_without_replacement,
        )
    elif sampling_mode == "coverage":
        packed = _build_coverage_packed_indices(
            point_positions=point_positions,
            sp_centroids=sp_centroids,
            super_index=super_index,
            counts=counts,
            ptr=ptr,
            superpoint_indices=superpoint_indices,
            n_sample=n_sample,
            sampling_without_replacement=sampling_without_replacement,
        )
        if packed is not None:
            packed_point_idx, packed_mask = packed
        else:
            packed_point_idx = torch.full((num_superpoints, n_sample), -1, dtype=torch.long, device=device)
            packed_mask = torch.zeros((num_superpoints, n_sample), dtype=torch.bool, device=device)

            for row, sp_idx in enumerate(superpoint_indices.tolist()):
                start = int(ptr[sp_idx].item())
                end = int(ptr[sp_idx + 1].item())
                if end <= start:
                    continue

                group_idx = order[start:end]
                centroid = point_positions[group_idx].mean(dim=0)
                selected_local_idx, selected_mask = _select_point_samples(
                    point_positions=point_positions[group_idx],
                    point_features=raw_data[group_idx],
                    centroid=centroid,
                    n_sample=n_sample,
                    sampling_mode=sampling_mode,
                    sampling_without_replacement=sampling_without_replacement,
                )

                valid = selected_mask & (selected_local_idx >= 0)
                if valid.any():
                    packed_point_idx[row, valid] = group_idx[selected_local_idx[valid]]
                    packed_mask[row, valid] = True
    else:
        packed_point_idx = torch.full((num_superpoints, n_sample), -1, dtype=torch.long, device=device)
        packed_mask = torch.zeros((num_superpoints, n_sample), dtype=torch.bool, device=device)

        for row, sp_idx in enumerate(superpoint_indices.tolist()):
            start = int(ptr[sp_idx].item())
            end = int(ptr[sp_idx + 1].item())
            if end <= start:
                continue

            group_idx = order[start:end]
            centroid = sp_centroids[sp_idx] if sp_centroids is not None else point_positions[group_idx].mean(dim=0)
            selected_local_idx, selected_mask = _select_point_samples(
                point_positions=point_positions[group_idx],
                point_features=raw_data[group_idx],
                centroid=centroid,
                n_sample=n_sample,
                sampling_mode=sampling_mode,
                sampling_without_replacement=sampling_without_replacement,
            )

            valid = selected_mask & (selected_local_idx >= 0)
            if valid.any():
                packed_point_idx[row, valid] = group_idx[selected_local_idx[valid]]
                packed_mask[row, valid] = True

    safe_idx = packed_point_idx.clamp(min=0)
    packed_raw_points = raw_data[safe_idx.reshape(-1)].view(num_superpoints, n_sample, raw_data.shape[-1])
    packed_raw_points[~packed_mask] = 0.0
    return packed_raw_points, packed_point_idx, packed_mask


def build_candidate_point_cloud(
    nag,
    raw_keys: Optional[List[str]],
    candidate_indices: torch.Tensor,
    device: Optional[torch.device] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Gather all level-0 points that belong to selected candidate superpoints.

    Returns:
        point_indices: level-0 point indices in the original scene.
        candidate_rows: row indices into candidate_indices for each gathered point.
        raw_points: concatenated raw attributes for each gathered point.
    """
    if raw_keys is None:
        raw_keys = ["pos"]
    if device is None:
        device = nag[0].pos.device

    super_index = getattr(nag[0], "super_index", None)
    if super_index is None:
        raise ValueError("NAG level-0 data must provide super_index for BSR point propagation")
    super_index = super_index.to(device=device, dtype=torch.long)
    candidate_indices = candidate_indices.to(device=device, dtype=torch.long)

    raw_tensors = []
    for key in raw_keys:
        if hasattr(nag[0], key):
            attr = getattr(nag[0], key)
            if attr is not None:
                if attr.dim() == 1:
                    attr = attr.unsqueeze(-1)
                raw_tensors.append(attr.to(device, non_blocking=True))
    if not raw_tensors:
        raise ValueError(f"Could not gather raw keys for BSR point propagation: {raw_keys}")

    raw_data = torch.cat(raw_tensors, dim=-1) if len(raw_tensors) > 1 else raw_tensors[0]
    if candidate_indices.numel() == 0 or super_index.numel() == 0:
        return (
            torch.empty(0, dtype=torch.long, device=device),
            torch.empty(0, dtype=torch.long, device=device),
            raw_data.new_zeros((0, raw_data.shape[-1])),
        )

    num_superpoints = getattr(nag[1], "num_nodes", None)
    if num_superpoints is None:
        num_superpoints = nag[1].pos.shape[0]
    num_superpoints = int(num_superpoints)
    row_map = torch.full((num_superpoints,), -1, dtype=torch.long, device=device)
    valid_candidates = candidate_indices[(candidate_indices >= 0) & (candidate_indices < num_superpoints)]
    if valid_candidates.numel() != candidate_indices.numel():
        raise ValueError("candidate_indices contain values outside level-1 superpoint range")
    row_map[candidate_indices] = torch.arange(candidate_indices.numel(), device=device)

    valid_super = (super_index >= 0) & (super_index < num_superpoints)
    candidate_rows = torch.full_like(super_index, -1)
    candidate_rows[valid_super] = row_map[super_index[valid_super]]
    point_mask = candidate_rows >= 0
    point_indices = torch.arange(super_index.numel(), device=device, dtype=torch.long)[point_mask]
    candidate_rows = candidate_rows[point_mask]
    return point_indices, candidate_rows, raw_data[point_indices]


def run_bsr(
    selector: BoundaryPriorSelector,
    refiner: PointSuperpointMixtureRefiner,
    sp_features: torch.Tensor,
    sp_centroids: torch.Tensor,
    coarse_logits: torch.Tensor,
    packed_raw_points: torch.Tensor,
    packed_point_indices: Optional[torch.Tensor] = None,
    packed_point_mask: Optional[torch.Tensor] = None,
    handcrafted_features: Optional[torch.Tensor] = None,
    edge_index: Optional[torch.Tensor] = None,
    partition_adapter: Optional[OptionalPartitionAdapter] = None,
    sampling_mode: str = "coverage",
) -> BSROutput:
    score_terms = selector.compute_score_terms(
        coarse_logits=coarse_logits,
        handcrafted_features=handcrafted_features,
        edge_index=edge_index,
    )
    candidate_indices, total_scores = selector(
        coarse_logits=coarse_logits,
        handcrafted_features=handcrafted_features,
        edge_index=edge_index,
    )
    candidate_scores = total_scores[candidate_indices]
    candidate_score_terms = {
        key: value[candidate_indices]
        for key, value in score_terms.items()
        if key in selector.score_terms
    }

    candidate_raw_points = packed_raw_points[candidate_indices]
    candidate_point_indices = None
    candidate_point_mask = None
    if packed_point_indices is not None:
        candidate_point_indices = packed_point_indices[candidate_indices]
    if packed_point_mask is not None:
        candidate_point_mask = packed_point_mask[candidate_indices]

    refiner_output = refiner(
        candidate_indices=candidate_indices,
        all_sp_features=sp_features,
        all_sp_centroids=sp_centroids,
        candidate_raw_points=candidate_raw_points,
        candidate_point_mask=candidate_point_mask,
        base_logits=coarse_logits,
    )

    partition_output = None
    if partition_adapter is not None and candidate_indices.numel() > 0 and candidate_point_indices is not None:
        point_preds = refiner_output.sampled_point_logits.argmax(dim=-1)
        partition_output = partition_adapter(point_preds, candidate_point_indices, candidate_indices)

    candidate_ratio, num_valid_sampled_points, avg_valid_points_per_candidate, effective_refine_ratio = (
        _compute_output_diagnostics(
            num_superpoints=int(sp_features.shape[0]),
            num_candidates=int(candidate_indices.numel()),
            sampled_point_mask=candidate_point_mask,
            sampled_point_logits=refiner_output.sampled_point_logits,
        )
    )

    return BSROutput(
        candidate_indices=candidate_indices,
        candidate_scores=candidate_scores,
        refined_sp_features=refiner_output.refined_sp_features,
        refined_sp_logits=refiner_output.refined_sp_logits,
        sampled_point_logits=refiner_output.sampled_point_logits,
        coarse_candidate_logits=coarse_logits[candidate_indices].clone(),
        sampled_point_indices=candidate_point_indices,
        sampled_point_mask=candidate_point_mask,
        boundary_logits=refiner_output.boundary_logits,
        boundary_scores=refiner_output.boundary_scores,
        point_residual_gates=refiner_output.point_residual_gates,
        slot_logits=refiner_output.slot_logits,
        slot_assignments=refiner_output.slot_assignments,
        slot_masses=refiner_output.slot_masses,
        slot_tokens=refiner_output.slot_tokens,
        assignment_entropy=refiner_output.assignment_entropy,
        secondary_slot_mass=refiner_output.secondary_slot_mass,
        dual_slot_activation_ratio=refiner_output.dual_slot_activation_ratio,
        slot_diversity=refiner_output.slot_diversity,
        selector_score_terms=candidate_score_terms,
        selector_score_summary=_summarize_selector_terms(candidate_score_terms),
        num_superpoints=int(sp_features.shape[0]),
        num_candidates=int(candidate_indices.numel()),
        candidate_ratio=candidate_ratio,
        num_valid_sampled_points=num_valid_sampled_points,
        avg_valid_points_per_candidate=avg_valid_points_per_candidate,
        effective_refine_ratio=effective_refine_ratio,
        sampling_mode=sampling_mode,
        point_head_mode=getattr(refiner, "variant", None),
        refiner_variant=refiner_output.refiner_variant,
        partition_output=partition_output,
    )


class BSRModule(nn.Module):
    def __init__(
        self,
        d_model: int = 64,
        num_classes: int = 13,
        selector_topk_ratio: float = 0.2,
        selector_score_terms: Optional[List[str]] = None,
        selector_term_weights: Optional[dict] = None,
        selector_scatter_idx: int = 2,
        n_sample: int = 64,
        d_raw: int = 6,
        n_heads: int = 4,
        token_mode: str = "superpoint_query",
        dropout: float = 0.1,
        hidden_dim: Optional[int] = None,
        variant: str = "mixture",
        n_subregions: int = 2,
        assignment_temperature: float = 1.0,
        point_head_mode: Optional[str] = None,
        use_boundary_head: bool = True,
        partition_adapter_enable: bool = False,
        partition_adapter_min_points: int = 3,
        partition_adapter_merge_small: bool = True,
    ):
        super().__init__()
        self.n_sample = int(n_sample)
        self.selector = BoundaryPriorSelector(
            topk_ratio=selector_topk_ratio,
            score_terms=selector_score_terms,
            term_weights=selector_term_weights,
            scatter_idx=selector_scatter_idx,
        )
        self.refiner = PointSuperpointMixtureRefiner(
            d_model=d_model,
            num_classes=num_classes,
            d_raw=d_raw,
            n_heads=n_heads,
            dropout=dropout,
            token_mode=token_mode,
            hidden_dim=hidden_dim,
            variant=variant,
            n_subregions=n_subregions,
            assignment_temperature=assignment_temperature,
            point_head_mode=point_head_mode,
            use_boundary_head=use_boundary_head,
        )
        self.partition_adapter = OptionalPartitionAdapter(
            enable=partition_adapter_enable,
            min_points_per_sp=partition_adapter_min_points,
            merge_small_clusters=partition_adapter_merge_small,
        )

    def select_candidates(
        self,
        coarse_logits: torch.Tensor,
        handcrafted_features: Optional[torch.Tensor] = None,
        edge_index: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        score_terms = self.selector.compute_score_terms(
            coarse_logits=coarse_logits,
            handcrafted_features=handcrafted_features,
            edge_index=edge_index,
        )
        candidate_indices, total_scores = self.selector(
            coarse_logits=coarse_logits,
            handcrafted_features=handcrafted_features,
            edge_index=edge_index,
        )
        candidate_scores = total_scores[candidate_indices]
        candidate_score_terms = {
            key: value[candidate_indices]
            for key, value in score_terms.items()
            if key in self.selector.score_terms
        }
        return candidate_indices, candidate_scores, candidate_score_terms

    def refine_candidates(
        self,
        candidate_indices: torch.Tensor,
        candidate_scores: torch.Tensor,
        sp_features: torch.Tensor,
        sp_centroids: torch.Tensor,
        coarse_logits: torch.Tensor,
        candidate_raw_points: torch.Tensor,
        candidate_point_indices: Optional[torch.Tensor] = None,
        candidate_point_mask: Optional[torch.Tensor] = None,
        selector_score_terms: Optional[Dict[str, torch.Tensor]] = None,
        sampling_mode: str = "coverage",
    ) -> BSROutput:
        refiner_output = self.refiner(
            candidate_indices=candidate_indices,
            all_sp_features=sp_features,
            all_sp_centroids=sp_centroids,
            candidate_raw_points=candidate_raw_points,
            candidate_point_mask=candidate_point_mask,
            base_logits=coarse_logits,
        )

        partition_output = None
        if self.partition_adapter.enable and candidate_indices.numel() > 0 and candidate_point_indices is not None:
            point_preds = refiner_output.sampled_point_logits.argmax(dim=-1)
            partition_output = self.partition_adapter(point_preds, candidate_point_indices, candidate_indices)

        candidate_ratio, num_valid_sampled_points, avg_valid_points_per_candidate, effective_refine_ratio = (
            _compute_output_diagnostics(
                num_superpoints=int(sp_features.shape[0]),
                num_candidates=int(candidate_indices.numel()),
                sampled_point_mask=candidate_point_mask,
                sampled_point_logits=refiner_output.sampled_point_logits,
            )
        )

        return BSROutput(
            candidate_indices=candidate_indices,
            candidate_scores=candidate_scores,
            refined_sp_features=refiner_output.refined_sp_features,
            refined_sp_logits=refiner_output.refined_sp_logits,
            sampled_point_logits=refiner_output.sampled_point_logits,
            coarse_candidate_logits=coarse_logits[candidate_indices].clone(),
            sampled_point_indices=candidate_point_indices,
            sampled_point_mask=candidate_point_mask,
            boundary_logits=refiner_output.boundary_logits,
            boundary_scores=refiner_output.boundary_scores,
            point_residual_gates=refiner_output.point_residual_gates,
            slot_logits=refiner_output.slot_logits,
            slot_assignments=refiner_output.slot_assignments,
            slot_masses=refiner_output.slot_masses,
            slot_tokens=refiner_output.slot_tokens,
            assignment_entropy=refiner_output.assignment_entropy,
            secondary_slot_mass=refiner_output.secondary_slot_mass,
            dual_slot_activation_ratio=refiner_output.dual_slot_activation_ratio,
            slot_diversity=refiner_output.slot_diversity,
            selector_score_terms=selector_score_terms,
            selector_score_summary=_summarize_selector_terms(selector_score_terms),
            num_superpoints=int(sp_features.shape[0]),
            num_candidates=int(candidate_indices.numel()),
            candidate_ratio=candidate_ratio,
            num_valid_sampled_points=num_valid_sampled_points,
            avg_valid_points_per_candidate=avg_valid_points_per_candidate,
            effective_refine_ratio=effective_refine_ratio,
            sampling_mode=sampling_mode,
            point_head_mode=getattr(self.refiner, "variant", None),
            refiner_variant=refiner_output.refiner_variant,
            partition_output=partition_output,
        )

    def forward(
        self,
        sp_features: torch.Tensor,
        sp_centroids: torch.Tensor,
        coarse_logits: torch.Tensor,
        packed_raw_points: torch.Tensor,
        packed_point_indices: Optional[torch.Tensor] = None,
        packed_point_mask: Optional[torch.Tensor] = None,
        handcrafted_features: Optional[torch.Tensor] = None,
        edge_index: Optional[torch.Tensor] = None,
        sampling_mode: str = "coverage",
    ) -> BSROutput:
        candidate_indices, candidate_scores, selector_score_terms = self.select_candidates(
            coarse_logits=coarse_logits,
            handcrafted_features=handcrafted_features,
            edge_index=edge_index,
        )
        candidate_raw_points = packed_raw_points[candidate_indices]
        candidate_point_indices = (
            packed_point_indices[candidate_indices] if packed_point_indices is not None else None
        )
        candidate_point_mask = (
            packed_point_mask[candidate_indices] if packed_point_mask is not None else None
        )
        return self.refine_candidates(
            candidate_indices=candidate_indices,
            candidate_scores=candidate_scores,
            sp_features=sp_features,
            sp_centroids=sp_centroids,
            coarse_logits=coarse_logits,
            candidate_raw_points=candidate_raw_points,
            candidate_point_indices=candidate_point_indices,
            candidate_point_mask=candidate_point_mask,
            selector_score_terms=selector_score_terms,
            sampling_mode=sampling_mode,
        )


def _gather_point_targets(
    sampled_point_indices: torch.Tensor,
    gt_labels: torch.Tensor,
    sampled_point_mask: Optional[torch.Tensor],
    num_classes: int,
    ignore_index: int,
) -> torch.Tensor:
    if gt_labels.dim() == 2:
        gt_labels = gt_labels.squeeze(1) if gt_labels.shape[1] == 1 else gt_labels.argmax(dim=1)

    if gt_labels.numel() == 0:
        return torch.full_like(sampled_point_indices, ignore_index)

    max_gt_index = max(int(gt_labels.shape[0]) - 1, 0)
    safe_idx = sampled_point_indices.clamp(min=0, max=max_gt_index)
    sampled_targets = gt_labels[safe_idx.reshape(-1)].view_as(sampled_point_indices)

    invalid_index_mask = (sampled_point_indices < 0) | (sampled_point_indices >= gt_labels.shape[0])
    sampled_targets[invalid_index_mask] = ignore_index
    invalid_label_mask = (sampled_targets < 0) | (sampled_targets >= num_classes)
    sampled_targets[invalid_label_mask] = ignore_index
    if sampled_point_mask is not None:
        sampled_targets[~sampled_point_mask] = ignore_index
    return sampled_targets


def _masked_mean(values: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
    if mask is None:
        return values.mean(dim=1)
    valid = mask.unsqueeze(-1).to(values.dtype)
    count = valid.sum(dim=1).clamp(min=1.0)
    return (values * valid).sum(dim=1) / count


def _finite_or_zero(value: torch.Tensor, zero: torch.Tensor) -> torch.Tensor:
    if torch.isfinite(value).all():
        return value
    return zero.clone()


def compute_bsr_losses(
    bsr_output: BSROutput,
    gt_labels: torch.Tensor,
    num_classes: int,
    ignore_index: int,
    lambda_refine: float = 0.5,
    lambda_consistency: float = 0.1,
    lambda_diversity: float = 0.0,
    lambda_boundary: float = 0.0,
    score_weighting: bool = False,
    score_weight_gamma: float = 1.0,
) -> Tuple[torch.Tensor, dict]:
    device = bsr_output.refined_sp_logits.device
    zero = torch.tensor(0.0, device=device)

    sampled_point_logits = bsr_output.sampled_point_logits
    sampled_point_indices = bsr_output.sampled_point_indices
    sampled_point_mask = bsr_output.sampled_point_mask
    candidate_indices = bsr_output.candidate_indices

    if (
        sampled_point_logits is None
        or sampled_point_indices is None
        or sampled_point_logits.numel() == 0
        or candidate_indices.numel() == 0
    ):
        losses = {
            "refine_loss": zero,
            "consistency_loss": zero,
            "diversity_loss": zero,
            "boundary_loss": zero,
            "total_loss": zero,
        }
        return zero, losses

    point_targets = _gather_point_targets(
        sampled_point_indices=sampled_point_indices,
        gt_labels=gt_labels,
        sampled_point_mask=sampled_point_mask,
        num_classes=num_classes,
        ignore_index=ignore_index,
    )

    refine_logits = sampled_point_logits.to(dtype=torch.float32)
    refine_logits = torch.nan_to_num(refine_logits, nan=0.0, posinf=10.0, neginf=-10.0)
    valid_point_targets = point_targets != ignore_index
    if valid_point_targets.any():
        refine_loss = F.cross_entropy(
            refine_logits.reshape(-1, refine_logits.shape[-1]),
            point_targets.reshape(-1),
            ignore_index=ignore_index,
        )
    else:
        refine_loss = zero

    candidate_weights = None
    if (
        score_weighting
        and bsr_output.candidate_scores is not None
        and bsr_output.candidate_scores.numel() == candidate_indices.numel()
    ):
        candidate_weights = bsr_output.candidate_scores.clamp(min=0).pow(float(score_weight_gamma))
        if torch.isfinite(candidate_weights).all() and candidate_weights.sum() > 0:
            candidate_weights = candidate_weights / candidate_weights.sum()
        else:
            candidate_weights = None

    if (
        bsr_output.fused_candidate_logits is not None
        and bsr_output.fused_candidate_logits.shape[0] == candidate_indices.numel()
    ):
        candidate_decision_logits = bsr_output.fused_candidate_logits
    else:
        candidate_decision_logits = bsr_output.refined_sp_logits[candidate_indices]

    if (
        bsr_output.slot_logits is not None
        and bsr_output.slot_masses is not None
        and bsr_output.slot_logits.numel() > 0
        and bsr_output.slot_masses.numel() > 0
    ):
        pooled_point_logits = torch.sum(
            bsr_output.slot_masses.to(dtype=torch.float32).unsqueeze(-1)
            * bsr_output.slot_logits.to(dtype=torch.float32),
            dim=1,
        )
    else:
        pooled_point_logits = _masked_mean(refine_logits, sampled_point_mask)
    candidate_decision_logits = candidate_decision_logits.to(dtype=torch.float32)
    pooled_point_logits = pooled_point_logits.to(dtype=torch.float32)
    log_p = torch.nan_to_num(F.log_softmax(candidate_decision_logits, dim=-1), nan=-100.0, posinf=0.0, neginf=-100.0)
    log_q = torch.nan_to_num(F.log_softmax(pooled_point_logits, dim=-1), nan=-100.0, posinf=0.0, neginf=-100.0)
    p = log_p.exp().clamp(min=1e-6)
    q = log_q.exp().clamp(min=1e-6)
    per_candidate_consistency = 0.5 * (
        F.kl_div(log_p, q, reduction="none").sum(dim=-1)
        + F.kl_div(log_q, p, reduction="none").sum(dim=-1)
    )
    if candidate_weights is not None:
        consistency_loss = (per_candidate_consistency * candidate_weights).sum()
    else:
        consistency_loss = per_candidate_consistency.mean()
    consistency_loss = _finite_or_zero(consistency_loss, zero)

    diversity_loss = zero
    if (
        lambda_diversity > 0.0
        and bsr_output.slot_logits is not None
        and bsr_output.slot_logits.shape[1] > 1
    ):
        slot_logits = bsr_output.slot_logits.to(dtype=torch.float32)
        slot_masses = bsr_output.slot_masses
        if slot_masses is not None and slot_masses.shape[1] > 1:
            sorted_masses = torch.sort(slot_masses.to(dtype=torch.float32), dim=-1, descending=True).values
            dual_usage = (2.0 * sorted_masses[:, 1]).clamp(min=0.0, max=1.0)
        else:
            dual_usage = torch.ones(slot_logits.shape[0], device=device, dtype=torch.float32)

        pair_losses = []
        for i in range(slot_logits.shape[1]):
            for j in range(i + 1, slot_logits.shape[1]):
                cosine = F.cosine_similarity(slot_logits[:, i], slot_logits[:, j], dim=-1)
                pair_losses.append((0.5 * (1.0 + cosine)) * dual_usage)
        if pair_losses:
            pair_losses_tensor = torch.stack(pair_losses, dim=0).mean(dim=0)
            if candidate_weights is not None:
                diversity_loss = (pair_losses_tensor * candidate_weights.to(pair_losses_tensor.dtype)).sum()
            else:
                diversity_loss = pair_losses_tensor.mean()
    diversity_loss = _finite_or_zero(diversity_loss, zero)

    boundary_loss = zero
    if (
        lambda_boundary > 0.0
        and (bsr_output.boundary_logits is not None or bsr_output.boundary_scores is not None)
    ):
        valid_mask = point_targets != ignore_index
        if valid_mask.any():
            majority_labels = []
            for idx in range(point_targets.shape[0]):
                labels_i = point_targets[idx][valid_mask[idx]]
                if labels_i.numel() == 0:
                    majority_labels.append(ignore_index)
                else:
                    majority_labels.append(torch.mode(labels_i).values.item())
            majority_labels = torch.tensor(majority_labels, device=device, dtype=point_targets.dtype)
            boundary_targets = point_targets.ne(majority_labels.unsqueeze(1)) & valid_mask
            if bsr_output.boundary_logits is not None:
                boundary_logits = bsr_output.boundary_logits.to(dtype=torch.float32)
            else:
                boundary_logits = torch.logit(
                    bsr_output.boundary_scores.to(dtype=torch.float32).clamp(
                        min=1e-6,
                        max=1.0 - 1e-6,
                    )
                )
            boundary_logits = torch.nan_to_num(boundary_logits, nan=0.0, posinf=10.0, neginf=-10.0)
            boundary_loss = F.binary_cross_entropy_with_logits(
                boundary_logits[valid_mask],
                boundary_targets[valid_mask].to(boundary_logits.dtype),
            )
    boundary_loss = _finite_or_zero(boundary_loss, zero)

    total_loss = (
        lambda_refine * refine_loss
        + lambda_consistency * consistency_loss
        + lambda_diversity * diversity_loss
        + lambda_boundary * boundary_loss
    )
    refine_loss = _finite_or_zero(refine_loss, zero)
    total_loss = _finite_or_zero(total_loss, zero)
    losses = {
        "refine_loss": refine_loss,
        "consistency_loss": consistency_loss,
        "diversity_loss": diversity_loss,
        "boundary_loss": boundary_loss,
        "total_loss": total_loss,
    }
    return total_loss, losses
