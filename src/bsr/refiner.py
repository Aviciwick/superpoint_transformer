from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


__all__ = ["PointSuperpointMixtureRefiner", "PointSuperpointBoundaryRefiner", "RefinerOutput"]


def _masked_mean(values: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
    if mask is None:
        return values.mean(dim=1)
    valid = mask.unsqueeze(-1).to(values.dtype)
    denom = valid.sum(dim=1).clamp(min=1.0)
    return (values * valid).sum(dim=1) / denom


def _normalized_entropy(probabilities: torch.Tensor) -> torch.Tensor:
    if probabilities.shape[-1] <= 1:
        return torch.zeros(probabilities.shape[:-1], device=probabilities.device, dtype=probabilities.dtype)
    entropy = -(probabilities.clamp(min=1e-6) * probabilities.clamp(min=1e-6).log()).sum(dim=-1)
    return entropy / torch.log(
        torch.tensor(float(probabilities.shape[-1]), device=probabilities.device, dtype=probabilities.dtype)
    )


def _pairwise_slot_diversity(slot_tokens: torch.Tensor) -> torch.Tensor:
    num_slots = slot_tokens.shape[1]
    if num_slots <= 1:
        return torch.zeros(slot_tokens.shape[0], device=slot_tokens.device, dtype=slot_tokens.dtype)

    normalized = F.normalize(slot_tokens, dim=-1)
    diversities = []
    for i in range(num_slots):
        for j in range(i + 1, num_slots):
            cosine = (normalized[:, i] * normalized[:, j]).sum(dim=-1)
            diversities.append(0.5 * (1.0 - cosine))
    return torch.stack(diversities, dim=0).mean(dim=0)


def _broadcast_slots_for_points(point_tokens: torch.Tensor, slot_tensor: torch.Tensor) -> torch.Tensor:
    while slot_tensor.dim() < point_tokens.dim() + 1:
        slot_tensor = slot_tensor.unsqueeze(-3)
    return slot_tensor


@dataclass
class RefinerOutput:
    refined_sp_features: torch.Tensor
    refined_sp_logits: torch.Tensor
    sampled_point_logits: torch.Tensor
    boundary_logits: Optional[torch.Tensor]
    boundary_scores: Optional[torch.Tensor]
    point_residual_gates: Optional[torch.Tensor]
    slot_logits: torch.Tensor
    slot_assignments: torch.Tensor
    slot_masses: torch.Tensor
    slot_tokens: torch.Tensor
    assignment_entropy: torch.Tensor
    secondary_slot_mass: torch.Tensor
    dual_slot_activation_ratio: float
    slot_diversity: torch.Tensor
    refiner_variant: str


class PointSuperpointMixtureRefiner(nn.Module):
    """Point-superpoint refiner with a mixture official path and legacy single-slot variants.

    Official path:
        - `variant='mixture'`
        - each candidate superpoint uses K latent subregion queries
        - sampled points softly assign to latent slots
        - point logits are mixture-weighted residual corrections over a refined base semantic anchor

    Ablation paths:
        - `variant='single_residual_gated'`
        - `variant='single_direct'`
    """

    _LEGACY_POINT_HEAD_MAP = {
        "residual_gated": "single_residual_gated",
        "direct": "single_direct",
    }

    def __init__(
        self,
        d_model: int = 64,
        num_classes: int = 13,
        d_raw: int = 6,
        n_heads: int = 4,
        dropout: float = 0.1,
        token_mode: str = "superpoint_query",
        hidden_dim: Optional[int] = None,
        variant: str = "mixture",
        n_subregions: int = 2,
        assignment_temperature: float = 1.0,
        point_head_mode: Optional[str] = None,
        use_boundary_head: bool = True,
    ):
        super().__init__()
        if token_mode != "superpoint_query":
            raise ValueError("token_mode must be 'superpoint_query'")
        if d_model % n_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by n_heads ({n_heads})")

        if point_head_mode is not None:
            variant = self._LEGACY_POINT_HEAD_MAP.get(point_head_mode, point_head_mode)
        if variant not in {"mixture", "single_residual_gated", "single_direct"}:
            raise ValueError(
                "variant must be one of {'mixture', 'single_residual_gated', 'single_direct'}"
            )
        if variant == "mixture" and int(n_subregions) < 2:
            raise ValueError("mixture variant requires n_subregions >= 2")

        hidden_dim = hidden_dim or max(d_model * 2, 128)
        self.d_model = int(d_model)
        self.num_classes = int(num_classes)
        self.token_mode = token_mode
        self.variant = str(variant)
        self.n_subregions = int(n_subregions if self.variant == "mixture" else 1)
        self.assignment_temperature = float(max(assignment_temperature, 1e-6))
        self.use_boundary_head = bool(use_boundary_head)
        self.configured_d_raw = int(d_raw)
        self._materialized_d_raw = None

        self.point_encoder = nn.Sequential(
            nn.LazyLinear(d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.query_proj = nn.Linear(d_model, d_model)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.refine_norm = nn.LayerNorm(d_model)
        self.refine_dropout = nn.Dropout(dropout)

        self.superpoint_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes),
        )

        conditioned_dim = d_model * 2
        if self.variant == "mixture":
            self.slot_embeddings = nn.Parameter(torch.randn(self.n_subregions, d_model) * 0.02)
            self.assignment_point_proj = nn.Linear(d_model, d_model)
            self.assignment_slot_proj = nn.Linear(d_model, d_model)
            self.slot_residual_head = nn.Sequential(
                nn.Linear(d_model, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, num_classes),
            )
            self.slot_gate_head = nn.Sequential(
                nn.Linear(d_model, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 1),
            )
        elif self.variant == "single_direct":
            self.point_head = nn.Sequential(
                nn.Linear(conditioned_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, num_classes),
            )
        else:
            self.point_residual_head = nn.Sequential(
                nn.Linear(conditioned_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, num_classes),
            )
            self.point_gate_head = nn.Sequential(
                nn.Linear(conditioned_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 1),
            )

        if self.use_boundary_head:
            self.boundary_head = nn.Sequential(
                nn.Linear(conditioned_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 1),
            )

    def _empty_output(
        self,
        all_sp_features: torch.Tensor,
        base_logits: torch.Tensor,
        num_points: int,
        num_candidates: int,
    ) -> RefinerOutput:
        device = all_sp_features.device
        dtype = all_sp_features.dtype
        num_slots = self.n_subregions if self.variant == "mixture" else 1
        empty_points = torch.zeros(num_candidates, num_points, self.num_classes, device=device, dtype=dtype)
        empty_boundary_logits = torch.zeros(num_candidates, num_points, device=device, dtype=dtype)
        empty_boundary = torch.zeros(num_candidates, num_points, device=device, dtype=dtype)
        empty_slot_logits = torch.zeros(num_candidates, num_slots, self.num_classes, device=device, dtype=dtype)
        empty_slot_assignments = torch.zeros(num_candidates, num_points, num_slots, device=device, dtype=dtype)
        empty_slot_masses = torch.zeros(num_candidates, num_slots, device=device, dtype=dtype)
        empty_entropy = torch.zeros(num_candidates, device=device, dtype=dtype)
        empty_secondary = torch.zeros(num_candidates, device=device, dtype=dtype)
        empty_diversity = torch.zeros(num_candidates, device=device, dtype=dtype)
        point_gates = torch.zeros(num_candidates, num_points, device=device, dtype=dtype)
        return RefinerOutput(
            refined_sp_features=all_sp_features.clone(),
            refined_sp_logits=base_logits.clone(),
            sampled_point_logits=empty_points,
            boundary_logits=empty_boundary_logits if self.use_boundary_head else None,
            boundary_scores=empty_boundary if self.use_boundary_head else None,
            point_residual_gates=point_gates,
            slot_logits=empty_slot_logits,
            slot_assignments=empty_slot_assignments,
            slot_masses=empty_slot_masses,
            slot_tokens=torch.zeros(num_candidates, num_slots, self.d_model, device=device, dtype=dtype),
            assignment_entropy=empty_entropy,
            secondary_slot_mass=empty_secondary,
            dual_slot_activation_ratio=0.0,
            slot_diversity=empty_diversity,
            refiner_variant=self.variant,
        )

    def _check_d_raw(self, actual_d_raw: int) -> None:
        encoder_layer = self.point_encoder[0]
        has_uninitialized = getattr(encoder_layer, "has_uninitialized_params", None)
        encoder_is_materialized = not has_uninitialized() if callable(has_uninitialized) else True
        if encoder_is_materialized:
            encoder_weight = getattr(encoder_layer, "weight", None)
            if torch.is_tensor(encoder_weight) and encoder_weight.ndim == 2:
                encoder_in_features = int(encoder_weight.shape[1])
            else:
                encoder_in_features = int(getattr(encoder_layer, "in_features", actual_d_raw))
            if encoder_in_features != actual_d_raw:
                raise RuntimeError(
                    f"BSR point_encoder expects d_raw={encoder_in_features}, "
                    f"but current input uses d_raw={actual_d_raw}. "
                    "Check raw_keys or checkpoint compatibility."
                )
        if self._materialized_d_raw is None:
            self._materialized_d_raw = int(actual_d_raw)
        elif int(actual_d_raw) != self._materialized_d_raw:
            raise RuntimeError(
                f"BSR point_encoder was materialized with d_raw={self._materialized_d_raw}, "
                f"but current input uses d_raw={actual_d_raw}. "
                "Check raw_keys or checkpoint compatibility."
            )

    def encode_candidate_points(
        self,
        raw_points: torch.Tensor,
        candidate_centroids: torch.Tensor,
    ) -> torch.Tensor:
        """Encode canonicalized raw points with XYZ expressed relative to candidate centroids."""
        if raw_points.shape[-1] < 3:
            raise ValueError(
                f"raw_points must contain XYZ in the first 3 channels, got D_raw={raw_points.shape[-1]}"
            )
        self._check_d_raw(int(raw_points.shape[-1]))

        raw_xyz = raw_points[..., :3]
        raw_attr = raw_points[..., 3:]
        if raw_points.dim() == 2:
            if candidate_centroids.dim() == 1:
                centroid = candidate_centroids.unsqueeze(0).expand_as(raw_xyz)
            elif candidate_centroids.shape == raw_xyz.shape:
                centroid = candidate_centroids
            else:
                raise ValueError("candidate_centroids must be [3] or [N, 3] for flat raw_points")
            local_xyz = raw_xyz - centroid
        elif raw_points.dim() == 3:
            if candidate_centroids.dim() == 2 and candidate_centroids.shape[0] == raw_points.shape[0]:
                centroid = candidate_centroids.unsqueeze(1)
            elif candidate_centroids.dim() == 3 and candidate_centroids.shape[:2] == raw_points.shape[:2]:
                centroid = candidate_centroids
            else:
                raise ValueError("candidate_centroids must be [K, 3] or [K, N, 3] for packed raw_points")
            local_xyz = raw_xyz - centroid
        else:
            raise ValueError("raw_points must be [N, D_raw] or [K, N, D_raw]")
        point_input = torch.cat([local_xyz, raw_attr], dim=-1)
        return self.point_encoder(point_input)

    def compute_slot_affinity(
        self,
        point_tokens: torch.Tensor,
        slot_tokens: torch.Tensor,
    ) -> torch.Tensor:
        """Compute point-to-slot affinity using the same projections as the sampled-point refiner."""
        if slot_tokens.shape[-2] <= 1 or self.variant != "mixture":
            return torch.ones(
                *point_tokens.shape[:-1],
                1,
                device=point_tokens.device,
                dtype=point_tokens.dtype,
            )

        slot_tokens = _broadcast_slots_for_points(point_tokens, slot_tokens)
        assign_point = F.normalize(self.assignment_point_proj(point_tokens), dim=-1)
        assign_slot = F.normalize(self.assignment_slot_proj(slot_tokens), dim=-1)
        assignment_logits = torch.einsum("...d,...sd->...s", assign_point, assign_slot)
        assignment_logits = assignment_logits / self.assignment_temperature
        return torch.softmax(assignment_logits, dim=-1)

    def propagate_slot_logits(
        self,
        point_tokens: torch.Tensor,
        slot_tokens: torch.Tensor,
        slot_logits: torch.Tensor,
    ) -> tuple:
        """Propagate slot logits to arbitrary candidate points through point-slot affinity."""
        if self.variant != "mixture":
            context = slot_tokens.squeeze(-2)
            if point_tokens.dim() == 3:
                context = context.unsqueeze(1).expand(-1, point_tokens.shape[1], -1)
            point_head_input = torch.cat([point_tokens, context], dim=-1)
            if self.variant == "single_direct":
                point_logits = self.point_head(point_head_input)
            else:
                residual_logits = self.point_residual_head(point_head_input)
                gates = torch.sigmoid(self.point_gate_head(point_head_input).squeeze(-1))
                base_logits = slot_logits.squeeze(-2)
                if point_tokens.dim() == 3:
                    base_logits = base_logits.unsqueeze(1)
                point_logits = base_logits + gates.unsqueeze(-1) * residual_logits
            slot_affinity = self.compute_slot_affinity(point_tokens, slot_tokens)
            return point_logits, slot_affinity

        slot_logits = _broadcast_slots_for_points(point_tokens, slot_logits)
        slot_affinity = self.compute_slot_affinity(point_tokens, slot_tokens)
        point_logits = torch.einsum("...s,...sc->...c", slot_affinity, slot_logits)
        return point_logits, slot_affinity

    def _prepare_candidate_points(
        self,
        candidate_indices: torch.Tensor,
        all_sp_features: torch.Tensor,
        all_sp_centroids: torch.Tensor,
        candidate_raw_points: torch.Tensor,
        candidate_point_mask: Optional[torch.Tensor],
    ):
        num_superpoints = all_sp_features.shape[0]
        num_candidates = candidate_indices.numel()
        if candidate_raw_points.shape[0] == num_superpoints:
            candidate_points = candidate_raw_points[candidate_indices]
        elif candidate_raw_points.shape[0] == num_candidates:
            candidate_points = candidate_raw_points
        else:
            raise ValueError("candidate_raw_points must be either [M, N, D_raw] or [K, N, D_raw]")

        self._check_d_raw(int(candidate_points.shape[-1]))

        candidate_features = all_sp_features[candidate_indices]
        candidate_centroids = all_sp_centroids[candidate_indices]
        point_tokens = self.encode_candidate_points(candidate_points, candidate_centroids)

        key_padding_mask = None
        if candidate_point_mask is not None:
            if candidate_point_mask.shape[0] == num_superpoints:
                candidate_point_mask = candidate_point_mask[candidate_indices]
            elif candidate_point_mask.shape[0] != num_candidates:
                raise ValueError("candidate_point_mask must be either [M, N] or [K, N]")
            key_padding_mask = ~candidate_point_mask.to(device=all_sp_features.device, dtype=torch.bool)

        return candidate_features, point_tokens, candidate_point_mask, key_padding_mask

    def _run_single_variant(
        self,
        candidate_features: torch.Tensor,
        point_tokens: torch.Tensor,
        candidate_point_mask: Optional[torch.Tensor],
        key_padding_mask: Optional[torch.Tensor],
        base_logits: torch.Tensor,
    ):
        query = self.query_proj(candidate_features).unsqueeze(1)
        refined_context, _ = self.cross_attn(
            query=query,
            key=point_tokens,
            value=point_tokens,
            key_padding_mask=key_padding_mask,
        )
        refined_candidate_features = self.refine_norm(
            candidate_features + self.refine_dropout(refined_context.squeeze(1))
        )
        refined_candidate_logits = self.superpoint_head(refined_candidate_features)
        point_context = refined_candidate_features.unsqueeze(1).expand(-1, point_tokens.shape[1], -1)
        point_head_input = torch.cat([point_tokens, point_context], dim=-1)

        point_residual_gates = None
        if self.variant == "single_direct":
            sampled_point_logits = self.point_head(point_head_input)
        else:
            residual_logits = self.point_residual_head(point_head_input)
            point_residual_gates = torch.sigmoid(self.point_gate_head(point_head_input).squeeze(-1))
            sampled_point_logits = (
                refined_candidate_logits.unsqueeze(1)
                + point_residual_gates.unsqueeze(-1) * residual_logits
            )

        boundary_logits = None
        boundary_scores = None
        if self.use_boundary_head:
            boundary_logits = self.boundary_head(point_head_input).squeeze(-1)
            boundary_scores = torch.sigmoid(boundary_logits)

        num_candidates, num_points = sampled_point_logits.shape[:2]
        slot_logits = refined_candidate_logits.unsqueeze(1)
        slot_assignments = torch.ones(
            num_candidates,
            num_points,
            1,
            device=sampled_point_logits.device,
            dtype=sampled_point_logits.dtype,
        )
        if candidate_point_mask is not None:
            slot_assignments = slot_assignments * candidate_point_mask.unsqueeze(-1).to(slot_assignments.dtype)
        slot_masses = torch.ones(num_candidates, 1, device=sampled_point_logits.device, dtype=sampled_point_logits.dtype)
        slot_tokens = refined_candidate_features.unsqueeze(1)
        assignment_entropy = torch.zeros(num_candidates, device=sampled_point_logits.device, dtype=sampled_point_logits.dtype)
        secondary_slot_mass = torch.zeros_like(assignment_entropy)
        slot_diversity = torch.zeros_like(assignment_entropy)

        return (
            refined_candidate_features,
            refined_candidate_logits,
            sampled_point_logits,
            boundary_logits,
            boundary_scores,
            point_residual_gates,
            slot_logits,
            slot_assignments,
            slot_masses,
            slot_tokens,
            assignment_entropy,
            secondary_slot_mass,
            0.0,
            slot_diversity,
        )

    def _run_mixture_variant(
        self,
        candidate_features: torch.Tensor,
        point_tokens: torch.Tensor,
        candidate_point_mask: Optional[torch.Tensor],
        key_padding_mask: Optional[torch.Tensor],
    ):
        num_candidates, num_points = point_tokens.shape[:2]
        slot_queries = candidate_features.unsqueeze(1) + self.slot_embeddings.unsqueeze(0)
        slot_queries = self.query_proj(slot_queries)
        slot_context, _ = self.cross_attn(
            query=slot_queries,
            key=point_tokens,
            value=point_tokens,
            key_padding_mask=key_padding_mask,
        )
        slot_context = torch.nan_to_num(slot_context, nan=0.0, posinf=10.0, neginf=-10.0)
        slot_tokens = self.refine_norm(slot_queries + self.refine_dropout(slot_context))

        point_proj = torch.nan_to_num(self.assignment_point_proj(point_tokens), nan=0.0, posinf=10.0, neginf=-10.0)
        slot_proj = torch.nan_to_num(self.assignment_slot_proj(slot_tokens), nan=0.0, posinf=10.0, neginf=-10.0)
        assign_point = F.normalize(point_proj, dim=-1, eps=1e-6)
        assign_slot = F.normalize(slot_proj, dim=-1, eps=1e-6)
        assignment_logits = torch.einsum("knd,ksd->kns", assign_point, assign_slot)
        assignment_logits = assignment_logits / self.assignment_temperature
        slot_assignments = torch.softmax(assignment_logits, dim=-1)
        if candidate_point_mask is not None:
            slot_assignments = slot_assignments * candidate_point_mask.unsqueeze(-1).to(slot_assignments.dtype)

        base_candidate_feature = slot_tokens.mean(dim=1)
        base_candidate_logits = self.superpoint_head(base_candidate_feature)
        slot_residuals = self.slot_residual_head(slot_tokens)
        slot_gates = torch.sigmoid(self.slot_gate_head(slot_tokens).squeeze(-1))
        slot_logits = base_candidate_logits.unsqueeze(1) + slot_gates.unsqueeze(-1) * slot_residuals

        slot_masses_raw = _masked_mean(slot_assignments, candidate_point_mask)
        slot_mass_sums = slot_masses_raw.sum(dim=-1, keepdim=True)
        uniform_mass = torch.full_like(slot_masses_raw, 1.0 / float(self.n_subregions))
        slot_masses = torch.where(
            slot_mass_sums > 1e-6,
            slot_masses_raw / slot_mass_sums.clamp(min=1e-6),
            uniform_mass,
        )

        sampled_point_logits = torch.einsum("kns,ksc->knc", slot_assignments, slot_logits)
        point_context = torch.einsum("kns,ksd->knd", slot_assignments, slot_tokens)
        point_residual_gates = torch.einsum("kns,ks->kn", slot_assignments, slot_gates)

        refined_candidate_features = torch.sum(slot_masses.unsqueeze(-1) * slot_tokens, dim=1)
        refined_candidate_logits = torch.sum(slot_masses.unsqueeze(-1) * slot_logits, dim=1)

        boundary_logits = None
        boundary_scores = None
        if self.use_boundary_head:
            boundary_input = torch.cat([point_tokens, point_context], dim=-1)
            boundary_logits = self.boundary_head(boundary_input).squeeze(-1)
            boundary_scores = torch.sigmoid(boundary_logits)

        entropy_per_point = _normalized_entropy(slot_assignments)
        assignment_entropy = _masked_mean(entropy_per_point.unsqueeze(-1), candidate_point_mask).squeeze(-1)
        sorted_masses = torch.sort(slot_masses, dim=-1, descending=True).values
        secondary_slot_mass = sorted_masses[:, 1] if self.n_subregions > 1 else torch.zeros_like(assignment_entropy)
        dual_slot_activation_ratio = float((secondary_slot_mass > 0.2).float().mean().item()) if num_candidates > 0 else 0.0
        slot_diversity = _pairwise_slot_diversity(slot_tokens)

        return (
            refined_candidate_features,
            refined_candidate_logits,
            sampled_point_logits,
            boundary_logits,
            boundary_scores,
            point_residual_gates,
            slot_logits,
            slot_assignments,
            slot_masses,
            slot_tokens,
            assignment_entropy,
            secondary_slot_mass,
            dual_slot_activation_ratio,
            slot_diversity,
        )

    def forward(
        self,
        candidate_indices: torch.Tensor,
        all_sp_features: torch.Tensor,
        all_sp_centroids: torch.Tensor,
        candidate_raw_points: torch.Tensor,
        candidate_point_mask: Optional[torch.Tensor],
        base_logits: torch.Tensor,
    ) -> RefinerOutput:
        num_superpoints = all_sp_features.shape[0]
        device = all_sp_features.device
        dtype = all_sp_features.dtype
        num_points = candidate_raw_points.shape[1] if candidate_raw_points.dim() == 3 else 0

        if candidate_raw_points.dim() != 3:
            raise ValueError(
                "candidate_raw_points must be a 3D tensor shaped [M, N, D_raw] or [K, N, D_raw]"
            )
        if candidate_raw_points.shape[-1] < 3:
            raise ValueError(
                f"candidate_raw_points must contain XYZ in the first 3 channels, got D_raw={candidate_raw_points.shape[-1]}"
            )

        if num_superpoints == 0:
            return self._empty_output(
                all_sp_features=all_sp_features,
                base_logits=base_logits,
                num_points=0,
                num_candidates=0,
            )

        num_candidates = candidate_indices.numel()
        if num_candidates == 0:
            return self._empty_output(
                all_sp_features=all_sp_features,
                base_logits=base_logits,
                num_points=num_points,
                num_candidates=0,
            )

        candidate_features, point_tokens, candidate_point_mask, key_padding_mask = self._prepare_candidate_points(
            candidate_indices=candidate_indices,
            all_sp_features=all_sp_features,
            all_sp_centroids=all_sp_centroids,
            candidate_raw_points=candidate_raw_points,
            candidate_point_mask=candidate_point_mask,
        )

        if self.variant == "mixture":
            (
                refined_candidate_features,
                refined_candidate_logits,
                sampled_point_logits,
                boundary_logits,
                boundary_scores,
                point_residual_gates,
                slot_logits,
                slot_assignments,
                slot_masses,
                slot_tokens,
                assignment_entropy,
                secondary_slot_mass,
                dual_slot_activation_ratio,
                slot_diversity,
            ) = self._run_mixture_variant(
                candidate_features=candidate_features,
                point_tokens=point_tokens,
                candidate_point_mask=candidate_point_mask,
                key_padding_mask=key_padding_mask,
            )
        else:
            (
                refined_candidate_features,
                refined_candidate_logits,
                sampled_point_logits,
                boundary_logits,
                boundary_scores,
                point_residual_gates,
                slot_logits,
                slot_assignments,
                slot_masses,
                slot_tokens,
                assignment_entropy,
                secondary_slot_mass,
                dual_slot_activation_ratio,
                slot_diversity,
            ) = self._run_single_variant(
                candidate_features=candidate_features,
                point_tokens=point_tokens,
                candidate_point_mask=candidate_point_mask,
                key_padding_mask=key_padding_mask,
                base_logits=base_logits,
            )

        refined_sp_features = all_sp_features.clone()
        refined_sp_features[candidate_indices] = refined_candidate_features.to(refined_sp_features.dtype)
        refined_sp_logits = base_logits.clone()
        refined_sp_logits[candidate_indices] = refined_candidate_logits.to(refined_sp_logits.dtype)

        if candidate_point_mask is not None:
            sampled_point_logits = sampled_point_logits.clone()
            sampled_point_logits[~candidate_point_mask] = 0.0
            if boundary_logits is not None:
                boundary_logits = boundary_logits.clone()
                boundary_logits[~candidate_point_mask] = 0.0
            if boundary_scores is not None:
                boundary_scores = boundary_scores.clone()
                boundary_scores[~candidate_point_mask] = 0.0
            if point_residual_gates is not None:
                point_residual_gates = point_residual_gates.clone()
                point_residual_gates[~candidate_point_mask] = 0.0

        return RefinerOutput(
            refined_sp_features=refined_sp_features,
            refined_sp_logits=refined_sp_logits,
            sampled_point_logits=sampled_point_logits,
            boundary_logits=boundary_logits,
            boundary_scores=boundary_scores,
            point_residual_gates=point_residual_gates,
            slot_logits=slot_logits,
            slot_assignments=slot_assignments,
            slot_masses=slot_masses,
            slot_tokens=slot_tokens,
            assignment_entropy=assignment_entropy,
            secondary_slot_mass=secondary_slot_mass,
            dual_slot_activation_ratio=dual_slot_activation_ratio,
            slot_diversity=slot_diversity,
            refiner_variant=self.variant,
        )


# Backward-compatible alias used by existing imports.
PointSuperpointBoundaryRefiner = PointSuperpointMixtureRefiner
