import os
import sys

import pytest
import torch


project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, "src"))

from src.bsr import BoundaryPriorSelector


class TestBoundaryPriorSelector:
    @pytest.fixture
    def selector(self):
        return BoundaryPriorSelector(
            topk_ratio=0.5,
            score_terms=("boundary",),
            term_weights={"boundary": 1.0},
        )

    def test_boundary_proxy_selection(self, selector):
        coarse_logits = torch.tensor(
            [
                [6.0, -6.0],
                [-6.0, 6.0],
                [6.0, -6.0],
                [6.0, -6.0],
            ],
            dtype=torch.float32,
        )
        edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long)

        candidate_indices, candidate_scores = selector(
            coarse_logits=coarse_logits,
            handcrafted_features=None,
            edge_index=edge_index,
        )

        assert candidate_scores.shape == (4,)
        assert candidate_indices.numel() == 2
        assert 1 in candidate_indices.tolist()

    def test_empty_superpoints(self, selector):
        coarse_logits = torch.zeros(0, 3)
        candidate_indices, candidate_scores = selector(coarse_logits)
        assert candidate_indices.numel() == 0
        assert candidate_scores.numel() == 0
