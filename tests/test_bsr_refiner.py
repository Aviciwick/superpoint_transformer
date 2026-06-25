import os
import sys
from unittest.mock import patch

import pytest
import torch


project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, "src"))

from src.bsr import PointSuperpointMixtureRefiner


class TestPointSuperpointMixtureRefiner:
    @pytest.fixture
    def module(self):
        return PointSuperpointMixtureRefiner(
            d_model=32,
            num_classes=5,
            d_raw=6,
            n_heads=4,
            token_mode="superpoint_query",
            variant="mixture",
            n_subregions=2,
        )

    @pytest.fixture
    def sample_data(self):
        torch.manual_seed(42)
        candidate_indices = torch.tensor([1, 3], dtype=torch.long)
        all_sp_features = torch.randn(5, 32)
        all_sp_centroids = torch.randn(5, 3)
        packed_raw_points = torch.randn(5, 16, 6)
        packed_mask = torch.ones(5, 16, dtype=torch.bool)
        base_logits = torch.randn(5, 5)
        return candidate_indices, all_sp_features, all_sp_centroids, packed_raw_points, packed_mask, base_logits

    def test_superpoint_query_mode_uses_two_slot_queries(self, module, sample_data):
        args = sample_data
        module.eval()
        with patch.object(module.cross_attn, "forward", wraps=module.cross_attn.forward) as mocked:
            with torch.no_grad():
                module(*args)

        query = mocked.call_args.kwargs["query"]
        assert query.shape[1] == module.n_subregions

    def test_mixture_output_shapes(self, module, sample_data):
        output = module(*sample_data)
        candidate_indices, all_sp_features, _, packed_raw_points, packed_mask, base_logits = sample_data

        assert output.refined_sp_features.shape == all_sp_features.shape
        assert output.refined_sp_logits.shape == base_logits.shape
        assert output.sampled_point_logits.shape == (
            candidate_indices.numel(),
            packed_raw_points.shape[1],
            base_logits.shape[1],
        )
        assert output.boundary_logits.shape == (candidate_indices.numel(), packed_raw_points.shape[1])
        assert output.boundary_scores.shape == (candidate_indices.numel(), packed_raw_points.shape[1])
        assert output.point_residual_gates.shape == (candidate_indices.numel(), packed_raw_points.shape[1])
        assert output.slot_logits.shape == (candidate_indices.numel(), 2, base_logits.shape[1])
        assert output.slot_assignments.shape == (candidate_indices.numel(), packed_raw_points.shape[1], 2)
        assert output.slot_masses.shape == (candidate_indices.numel(), 2)
        assert output.slot_tokens.shape == (candidate_indices.numel(), 2, module.d_model)
        assert output.assignment_entropy.shape == (candidate_indices.numel(),)
        assert output.secondary_slot_mass.shape == (candidate_indices.numel(),)
        assert output.slot_diversity.shape == (candidate_indices.numel(),)
        assert output.refiner_variant == "mixture"
        valid_assignments = output.slot_assignments[packed_mask[candidate_indices]]
        assert torch.allclose(
            valid_assignments.sum(dim=-1),
            torch.ones_like(valid_assignments[..., 0]),
            atol=1e-5,
        )
        assert torch.all(output.slot_masses >= 0.0)
        assert torch.allclose(output.slot_masses.sum(dim=-1), torch.ones(candidate_indices.numel()), atol=1e-5)
        assert torch.allclose(output.boundary_scores, torch.sigmoid(output.boundary_logits), atol=1e-6)

    def test_all_point_slot_propagation_allows_point_specific_logits(self, module):
        torch.manual_seed(123)
        point_tokens = torch.randn(4, module.d_model)
        slot_tokens = torch.randn(4, 2, module.d_model)
        slot_logits = torch.randn(4, 2, module.num_classes)

        point_logits, slot_affinity = module.propagate_slot_logits(
            point_tokens=point_tokens,
            slot_tokens=slot_tokens,
            slot_logits=slot_logits,
        )

        assert point_logits.shape == (4, module.num_classes)
        assert slot_affinity.shape == (4, 2)
        assert torch.allclose(slot_affinity.sum(dim=-1), torch.ones(4), atol=1e-5)
        assert not torch.allclose(point_logits[0], point_logits[1])

        packed_logits, packed_affinity = module.propagate_slot_logits(
            point_tokens=torch.randn(2, 3, module.d_model),
            slot_tokens=torch.randn(2, 2, module.d_model),
            slot_logits=torch.randn(2, 2, module.num_classes),
        )
        assert packed_logits.shape == (2, 3, module.num_classes)
        assert packed_affinity.shape == (2, 3, 2)

    def test_materialized_d_raw_mismatch_raises(self, module, sample_data):
        module(*sample_data)
        candidate_indices, all_sp_features, all_sp_centroids, packed_raw_points, packed_mask, base_logits = sample_data
        mismatched_raw_points = torch.randn(packed_raw_points.shape[0], packed_raw_points.shape[1], 9)
        with pytest.raises(RuntimeError, match="d_raw"):
            module(
                candidate_indices,
                all_sp_features,
                all_sp_centroids,
                mismatched_raw_points,
                packed_mask,
                base_logits,
            )

    def test_materialized_weight_shape_overrides_stale_in_features(self, module, sample_data):
        module(*sample_data)
        encoder_layer = module.point_encoder[0]
        encoder_layer.in_features = 0

        output = module(*sample_data)

        assert output.sampled_point_logits.shape[0] == sample_data[0].numel()

    def test_single_direct_variant_disables_gates_and_collapses_to_one_slot(self, sample_data):
        module = PointSuperpointMixtureRefiner(
            d_model=32,
            num_classes=5,
            d_raw=6,
            n_heads=4,
            token_mode="superpoint_query",
            variant="single_direct",
        )

        output = module(*sample_data)
        assert output.refiner_variant == "single_direct"
        assert output.point_residual_gates is None
        assert output.slot_logits.shape[1] == 1
        assert output.slot_assignments.shape[-1] == 1
        assert torch.allclose(output.slot_masses, torch.ones_like(output.slot_masses))

    def test_empty_candidate_path_returns_empty_tensors(self, module, sample_data):
        _, all_sp_features, all_sp_centroids, packed_raw_points, packed_mask, base_logits = sample_data
        empty_indices = torch.zeros(0, dtype=torch.long)
        output = module(
            empty_indices,
            all_sp_features,
            all_sp_centroids,
            packed_raw_points,
            packed_mask,
            base_logits,
        )

        assert output.sampled_point_logits.shape[0] == 0
        assert output.slot_logits.shape[0] == 0
        assert output.slot_assignments.shape[0] == 0
