import math
from typing import Dict, Iterable, Optional, Tuple

import torch
import torch.nn as nn


__all__ = ["BoundaryPriorSelector"]


class BoundaryPriorSelector(nn.Module):
    """Heuristic candidate selector for boundary-sensitive superpoints.

    The selector is intentionally lightweight. It provides a candidate prior
    for the learnable refiner rather than acting as the main innovation.
    """

    _VALID_TERMS = ("uncertainty", "geometry", "boundary")

    def __init__(
        self,
        topk_ratio: float = 0.2,
        score_terms: Optional[Iterable[str]] = None,
        term_weights: Optional[Dict[str, float]] = None,
        scatter_idx: int = 2,
        eps: float = 1e-6,
    ):
        super().__init__()
        if not 0.0 < topk_ratio <= 1.0:
            raise ValueError(f"topk_ratio must be in (0, 1], got {topk_ratio}")

        score_terms = tuple(score_terms or self._VALID_TERMS)
        invalid_terms = [term for term in score_terms if term not in self._VALID_TERMS]
        if invalid_terms:
            raise ValueError(f"Unsupported score terms: {invalid_terms}")

        self.topk_ratio = float(topk_ratio)
        self.score_terms = score_terms
        self.scatter_idx = int(scatter_idx)
        self.eps = float(eps)
        self.term_weights = dict(term_weights or {})

    def _score_dtype(self, coarse_logits: torch.Tensor) -> torch.dtype:
        # AMP often evaluates softmax-style scores in fp32 even when logits are fp16.
        # Keep the selector path in a consistent dtype so index_add_/accumulation do not
        # trip over Half/Float mismatches during training.
        if coarse_logits.dtype in (torch.float16, torch.bfloat16):
            return torch.float32
        return coarse_logits.dtype

    def _normalize(self, values: torch.Tensor) -> torch.Tensor:
        if values.numel() == 0:
            return values
        min_value = values.min()
        max_value = values.max()
        return (values - min_value) / (max_value - min_value + self.eps)

    def _uncertainty_score(self, coarse_logits: torch.Tensor) -> torch.Tensor:
        logits = coarse_logits.to(dtype=self._score_dtype(coarse_logits))
        num_classes = coarse_logits.shape[1]
        probs = torch.softmax(logits, dim=1)
        entropy = -(probs * torch.log(probs + self.eps)).sum(dim=1)
        entropy = torch.nan_to_num(entropy, nan=0.0, posinf=0.0, neginf=0.0)
        return entropy / (math.log(num_classes) + self.eps)

    def _geometry_score(
        self, coarse_logits: torch.Tensor, handcrafted_features: Optional[torch.Tensor]
    ) -> torch.Tensor:
        num_superpoints = coarse_logits.shape[0]
        score_dtype = self._score_dtype(coarse_logits)
        if handcrafted_features is None or handcrafted_features.numel() == 0:
            return torch.zeros(num_superpoints, device=coarse_logits.device, dtype=score_dtype)
        if handcrafted_features.shape[1] <= self.scatter_idx:
            return torch.zeros(num_superpoints, device=coarse_logits.device, dtype=score_dtype)
        values = handcrafted_features[:, self.scatter_idx].to(
            device=coarse_logits.device,
            dtype=score_dtype,
        )
        values = torch.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
        return self._normalize(values)

    def _boundary_score(
        self, coarse_logits: torch.Tensor, edge_index: Optional[torch.Tensor]
    ) -> torch.Tensor:
        num_superpoints = coarse_logits.shape[0]
        device = coarse_logits.device
        dtype = self._score_dtype(coarse_logits)
        if edge_index is None or edge_index.numel() == 0:
            return torch.zeros(num_superpoints, device=device, dtype=dtype)

        src = edge_index[0].to(device=device, dtype=torch.long)
        dst = edge_index[1].to(device=device, dtype=torch.long)
        valid = (src >= 0) & (src < num_superpoints) & (dst >= 0) & (dst < num_superpoints)
        if not valid.any():
            return torch.zeros(num_superpoints, device=device, dtype=dtype)

        src = src[valid]
        dst = dst[valid]
        probs = torch.softmax(coarse_logits.to(dtype=dtype), dim=1)
        edge_delta = (probs[src] - probs[dst]).abs().sum(dim=1)

        accum = torch.zeros(num_superpoints, device=device, dtype=dtype)
        counts = torch.zeros(num_superpoints, device=device, dtype=dtype)
        ones = torch.ones_like(edge_delta)
        accum.index_add_(0, src, edge_delta)
        accum.index_add_(0, dst, edge_delta)
        counts.index_add_(0, src, ones)
        counts.index_add_(0, dst, ones)
        proxy = accum / counts.clamp(min=1.0)
        proxy = torch.nan_to_num(proxy, nan=0.0, posinf=0.0, neginf=0.0)
        return self._normalize(proxy)

    def forward(
        self,
        coarse_logits: torch.Tensor,
        handcrafted_features: Optional[torch.Tensor] = None,
        edge_index: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if coarse_logits.dim() != 2:
            raise ValueError(f"coarse_logits must be [M, C], got {tuple(coarse_logits.shape)}")

        num_superpoints = coarse_logits.shape[0]
        if num_superpoints == 0:
            empty = torch.zeros(0, dtype=torch.long, device=coarse_logits.device)
            return empty, coarse_logits.new_zeros(0)

        scores = self.compute_score_terms(
            coarse_logits=coarse_logits,
            handcrafted_features=handcrafted_features,
            edge_index=edge_index,
        )

        active_weights = []
        for term in self.score_terms:
            active_weights.append(max(float(self.term_weights.get(term, 1.0)), 0.0))
        weight_sum = sum(active_weights) or 1.0

        score_dtype = next(iter(scores.values())).dtype if scores else self._score_dtype(coarse_logits)
        total_scores = torch.zeros(num_superpoints, device=coarse_logits.device, dtype=score_dtype)
        for term, raw_weight in zip(self.score_terms, active_weights):
            total_scores = total_scores + scores[term] * (raw_weight / weight_sum)

        num_candidates = min(num_superpoints, max(1, math.ceil(num_superpoints * self.topk_ratio)))
        _, candidate_indices = torch.topk(total_scores, k=num_candidates, largest=True, sorted=False)
        return candidate_indices, total_scores

    def compute_score_terms(
        self,
        coarse_logits: torch.Tensor,
        handcrafted_features: Optional[torch.Tensor] = None,
        edge_index: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        if coarse_logits.dim() != 2:
            raise ValueError(f"coarse_logits must be [M, C], got {tuple(coarse_logits.shape)}")

        return {
            "uncertainty": self._uncertainty_score(coarse_logits),
            "geometry": self._geometry_score(coarse_logits, handcrafted_features),
            "boundary": self._boundary_score(coarse_logits, edge_index),
        }
