import os
import sys

import pytest
import torch
import torch.nn.functional as F


project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, "src"))

from src.bsr import (
    BSRModule,
    BSROutput,
    build_candidate_point_cloud,
    build_packed_points,
    compute_bsr_losses,
)
from src.models.semantic import _resolve_available_bsr_raw_keys


class TestBSRIntegration:
    class DummyLevel:
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)

    def test_resolve_available_bsr_raw_keys_prefers_checkpoint_compatible_subset(self):
        level0 = self.DummyLevel(
            pos=torch.randn(8, 3),
            rgb=torch.randint(0, 255, (8, 3), dtype=torch.uint8),
        )

        resolved_pos_only = _resolve_available_bsr_raw_keys(
            level0,
            preferred_keys=["pos", "rgb"],
            expected_d_raw=3,
        )
        resolved_pos_rgb = _resolve_available_bsr_raw_keys(
            level0,
            preferred_keys=["pos", "rgb"],
            expected_d_raw=6,
        )

        assert resolved_pos_only == ["pos"]
        assert resolved_pos_rgb == ["pos", "rgb"]

    def test_bsr_module_forward_uses_mixture_variant(self):
        torch.manual_seed(7)
        module = BSRModule(
            d_model=32,
            num_classes=6,
            selector_topk_ratio=0.25,
            selector_score_terms=["uncertainty", "geometry", "boundary"],
            n_sample=8,
            d_raw=6,
            n_heads=4,
            variant="mixture",
            n_subregions=2,
        )

        sp_features = torch.randn(8, 32)
        sp_centroids = torch.randn(8, 3)
        coarse_logits = torch.randn(8, 6)
        packed_raw_points = torch.randn(8, 8, 6)
        packed_point_indices = torch.randint(0, 64, (8, 8))
        packed_point_mask = torch.ones(8, 8, dtype=torch.bool)
        handcrafted_features = torch.randn(8, 8)
        edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=torch.long)

        output = module(
            sp_features=sp_features,
            sp_centroids=sp_centroids,
            coarse_logits=coarse_logits,
            packed_raw_points=packed_raw_points,
            packed_point_indices=packed_point_indices,
            packed_point_mask=packed_point_mask,
            handcrafted_features=handcrafted_features,
            edge_index=edge_index,
        )

        assert isinstance(output, BSROutput)
        assert output.refined_sp_features.shape == sp_features.shape
        assert output.refined_sp_logits.shape == coarse_logits.shape
        assert output.candidate_indices.numel() == 2
        assert output.candidate_scores.shape == (2,)
        assert output.coarse_candidate_logits.shape == (2, 6)
        assert output.sampled_point_indices.shape == (2, 8)
        assert output.num_superpoints == 8
        assert output.num_candidates == 2
        assert output.candidate_ratio == pytest.approx(0.25)
        assert output.num_valid_sampled_points == 16
        assert output.avg_valid_points_per_candidate == pytest.approx(8.0)
        assert output.effective_refine_ratio == pytest.approx(0.25)
        assert output.selector_score_summary is not None
        assert set(output.selector_score_summary.keys()) == {"uncertainty", "geometry", "boundary"}
        assert output.refiner_variant == "mixture"
        assert output.slot_logits.shape == (2, 2, 6)
        assert output.slot_assignments.shape == (2, 8, 2)
        assert output.slot_masses.shape == (2, 2)
        assert output.slot_tokens.shape == (2, 2, 32)
        assert output.assignment_entropy.shape == (2,)
        assert output.secondary_slot_mass.shape == (2,)
        assert output.slot_diversity.shape == (2,)
        assert output.dual_slot_activation_ratio >= 0.0
        assert output.dual_slot_activation_ratio <= 1.0
        assert torch.allclose(output.slot_masses.sum(dim=-1), torch.ones(2), atol=1e-5)

    def test_build_candidate_point_cloud_gathers_all_candidate_points(self):
        level0 = self.DummyLevel(
            pos=torch.arange(18, dtype=torch.float32).view(6, 3),
            rgb=torch.ones(6, 3),
            super_index=torch.tensor([0, 1, 0, 2, 2, 3], dtype=torch.long),
        )
        level1 = self.DummyLevel(pos=torch.randn(4, 3), num_nodes=4)
        nag = [level0, level1]

        point_indices, candidate_rows, raw_points = build_candidate_point_cloud(
            nag=nag,
            raw_keys=["pos", "rgb"],
            candidate_indices=torch.tensor([0, 2], dtype=torch.long),
            device=torch.device("cpu"),
        )

        assert point_indices.tolist() == [0, 2, 3, 4]
        assert candidate_rows.tolist() == [0, 0, 1, 1]
        assert raw_points.shape == (4, 6)
        assert torch.allclose(raw_points[:, :3], level0.pos[point_indices])

    def test_single_residual_gated_variant_remains_available(self):
        torch.manual_seed(19)
        module = BSRModule(
            d_model=32,
            num_classes=4,
            selector_topk_ratio=0.5,
            n_sample=6,
            d_raw=6,
            n_heads=4,
            variant="single_residual_gated",
        )
        sp_features = torch.randn(6, 32)
        sp_centroids = torch.randn(6, 3)
        coarse_logits = torch.randn(6, 4)
        candidate_indices = torch.tensor([1, 4, 5], dtype=torch.long)
        candidate_scores = torch.rand(3)
        candidate_raw_points = torch.randn(3, 6, 6)
        candidate_point_indices = torch.randint(0, 50, (3, 6))
        candidate_point_mask = torch.ones(3, 6, dtype=torch.bool)
        score_terms = {
            "uncertainty": torch.rand(3),
            "geometry": torch.rand(3),
            "boundary": torch.rand(3),
        }

        output = module.refine_candidates(
            candidate_indices=candidate_indices,
            candidate_scores=candidate_scores,
            sp_features=sp_features,
            sp_centroids=sp_centroids,
            coarse_logits=coarse_logits,
            candidate_raw_points=candidate_raw_points,
            candidate_point_indices=candidate_point_indices,
            candidate_point_mask=candidate_point_mask,
            selector_score_terms=score_terms,
        )

        assert output.sampled_point_indices.shape == (3, 6)
        assert output.sampled_point_mask.shape == (3, 6)
        assert output.selector_score_terms["boundary"].shape == (3,)
        assert output.num_candidates == 3
        assert output.candidate_ratio == pytest.approx(0.5)
        assert output.num_valid_sampled_points == 18
        assert output.avg_valid_points_per_candidate == pytest.approx(6.0)
        assert output.effective_refine_ratio == pytest.approx(0.5)
        assert output.refiner_variant == "single_residual_gated"
        assert output.point_residual_gates is not None
        assert output.slot_logits.shape[1] == 1

    def test_bsr_losses_include_diversity_term(self):
        torch.manual_seed(11)
        candidate_indices = torch.tensor([0, 2], dtype=torch.long)
        sampled_point_indices = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.long)
        output = BSROutput(
            candidate_indices=candidate_indices,
            candidate_scores=torch.rand(2),
            refined_sp_features=torch.randn(4, 16),
            refined_sp_logits=torch.randn(4, 5),
            sampled_point_logits=torch.randn(2, 3, 5),
            fused_candidate_logits=torch.randn(2, 5),
            sampled_point_indices=sampled_point_indices,
            sampled_point_mask=torch.ones(2, 3, dtype=torch.bool),
            boundary_scores=torch.full((2, 3), 0.5),
            slot_logits=torch.randn(2, 2, 5),
            slot_masses=torch.tensor([[0.7, 0.3], [0.55, 0.45]], dtype=torch.float32),
        )
        gt_labels = torch.tensor([0, 1, 1, 2, 2, 3, 3, 4], dtype=torch.long)

        total_loss, loss_terms = compute_bsr_losses(
            bsr_output=output,
            gt_labels=gt_labels,
            num_classes=5,
            ignore_index=5,
            lambda_refine=0.5,
            lambda_consistency=0.1,
            lambda_diversity=0.2,
            lambda_boundary=0.2,
        )

        assert total_loss.ndim == 0
        assert not torch.isnan(total_loss)
        assert set(loss_terms.keys()) == {
            "refine_loss",
            "consistency_loss",
            "diversity_loss",
            "boundary_loss",
            "total_loss",
        }

    def test_bsr_losses_map_invalid_labels_and_indices_to_ignore(self):
        candidate_indices = torch.tensor([0], dtype=torch.long)
        sampled_point_logits = torch.tensor(
            [[
                [5.0, -2.0, -2.0],
                [0.2, 0.1, 0.0],
                [0.1, 0.3, 0.2],
                [0.1, 0.2, 0.3],
                [-2.0, 5.0, -2.0],
            ]],
            dtype=torch.float32,
        )
        output = BSROutput(
            candidate_indices=candidate_indices,
            candidate_scores=torch.tensor([1.0], dtype=torch.float32),
            refined_sp_features=torch.randn(1, 8),
            refined_sp_logits=torch.randn(1, 3),
            sampled_point_logits=sampled_point_logits,
            fused_candidate_logits=torch.randn(1, 3),
            sampled_point_indices=torch.tensor([[0, 1, 10, -1, 3]], dtype=torch.long),
            sampled_point_mask=torch.ones(1, 5, dtype=torch.bool),
            slot_logits=torch.randn(1, 1, 3),
            slot_masses=torch.ones(1, 1, dtype=torch.float32),
        )
        gt_labels = torch.tensor([0, -1, 2, 1], dtype=torch.long)

        total_loss, loss_terms = compute_bsr_losses(
            bsr_output=output,
            gt_labels=gt_labels,
            num_classes=3,
            ignore_index=3,
            lambda_refine=1.0,
            lambda_consistency=0.0,
            lambda_diversity=0.0,
            lambda_boundary=0.0,
        )

        expected_targets = torch.tensor([0, 3, 3, 3, 1], dtype=torch.long)
        expected_refine = F.cross_entropy(
            sampled_point_logits.view(-1, 3),
            expected_targets,
            ignore_index=3,
        )

        assert torch.allclose(loss_terms["refine_loss"], expected_refine)
        assert torch.allclose(total_loss, expected_refine)

    def test_bsr_losses_all_ignored_targets_stay_finite(self):
        output = BSROutput(
            candidate_indices=torch.tensor([0, 1], dtype=torch.long),
            candidate_scores=torch.tensor([0.8, 0.4], dtype=torch.float32),
            refined_sp_features=torch.randn(3, 8),
            refined_sp_logits=torch.randn(3, 3),
            sampled_point_logits=torch.randn(2, 4, 3),
            fused_candidate_logits=torch.randn(2, 3),
            sampled_point_indices=torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=torch.long),
            sampled_point_mask=torch.ones(2, 4, dtype=torch.bool),
            boundary_logits=torch.randn(2, 4),
            boundary_scores=torch.full((2, 4), 0.5, dtype=torch.float32),
            slot_logits=torch.randn(2, 2, 3),
            slot_masses=torch.tensor([[0.7, 0.3], [0.6, 0.4]], dtype=torch.float32),
        )
        gt_labels = torch.tensor([3, 3, 3, 3, 3], dtype=torch.long)

        total_loss, loss_terms = compute_bsr_losses(
            bsr_output=output,
            gt_labels=gt_labels,
            num_classes=3,
            ignore_index=3,
            lambda_refine=0.5,
            lambda_consistency=0.1,
            lambda_diversity=0.2,
            lambda_boundary=0.2,
            score_weighting=True,
        )

        assert torch.isfinite(total_loss)
        assert loss_terms["refine_loss"].item() == pytest.approx(0.0)
        assert loss_terms["boundary_loss"].item() == pytest.approx(0.0)
        assert torch.isfinite(loss_terms["consistency_loss"])
        assert torch.isfinite(loss_terms["diversity_loss"])

    def test_build_packed_points_boundary_hybrid_without_replacement(self):
        level0 = self.DummyLevel(
            pos=torch.tensor(
                [
                    [0.0, 0.0, 0.0],
                    [0.2, 0.0, 0.0],
                    [0.8, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.2, 1.0, 0.0],
                    [0.8, 1.0, 0.0],
                    [1.0, 1.0, 0.0],
                ],
                dtype=torch.float32,
            ),
            rgb=torch.tensor(
                [
                    [1.0, 0.0, 0.0],
                    [0.9, 0.0, 0.0],
                    [0.2, 0.0, 0.0],
                    [0.1, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.9, 0.0],
                    [0.0, 0.2, 0.0],
                    [0.0, 0.1, 0.0],
                ],
                dtype=torch.float32,
            ),
            super_index=torch.tensor([0, 0, 0, 0, 1, 1, 1, 1], dtype=torch.long),
        )
        level1 = self.DummyLevel(
            pos=torch.tensor([[0.5, 0.0, 0.0], [0.5, 1.0, 0.0]], dtype=torch.float32),
            num_nodes=2,
        )
        nag = [level0, level1]

        packed_raw_points, packed_point_idx, packed_mask = build_packed_points(
            nag,
            n_sample=3,
            raw_keys=["pos", "rgb"],
            sampling_mode="boundary_hybrid",
            sampling_without_replacement=True,
        )

        assert packed_raw_points.shape == (2, 3, 6)
        assert packed_point_idx.shape == (2, 3)
        assert packed_mask.all()
        for row in range(packed_point_idx.shape[0]):
            selected = packed_point_idx[row][packed_mask[row]]
            assert selected.unique().numel() == selected.numel()

    def test_build_packed_points_coverage_without_replacement(self):
        level0 = self.DummyLevel(
            pos=torch.tensor(
                [
                    [0.0, 0.0, 0.0],
                    [0.1, 0.0, 0.0],
                    [0.2, 0.0, 0.0],
                    [0.8, 0.0, 0.0],
                    [0.9, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                ],
                dtype=torch.float32,
            ),
            rgb=torch.tensor(
                [
                    [1.0, 0.0, 0.0],
                    [0.9, 0.0, 0.0],
                    [0.8, 0.0, 0.0],
                    [0.2, 0.0, 0.0],
                    [0.1, 0.0, 0.0],
                    [0.0, 0.0, 0.0],
                ],
                dtype=torch.float32,
            ),
            super_index=torch.tensor([0, 0, 0, 1, 1, 1], dtype=torch.long),
        )
        level1 = self.DummyLevel(
            pos=torch.tensor([[0.1, 0.0, 0.0], [0.9, 0.0, 0.0]], dtype=torch.float32),
            num_nodes=2,
        )
        nag = [level0, level1]

        _, packed_point_idx, packed_mask = build_packed_points(
            nag,
            n_sample=2,
            raw_keys=["pos", "rgb"],
            sampling_mode="coverage",
            sampling_without_replacement=True,
        )

        assert packed_mask.all()
        for row in range(packed_point_idx.shape[0]):
            selected = packed_point_idx[row][packed_mask[row]]
            assert selected.unique().numel() == selected.numel()

    def test_score_weighted_consistency_changes_total_loss(self):
        torch.manual_seed(5)
        candidate_indices = torch.tensor([0, 1], dtype=torch.long)
        sampled_point_indices = torch.tensor([[0, 1], [2, 3]], dtype=torch.long)
        output = BSROutput(
            candidate_indices=candidate_indices,
            candidate_scores=torch.tensor([1.0, 0.05], dtype=torch.float32),
            refined_sp_features=torch.randn(2, 8),
            refined_sp_logits=torch.tensor(
                [[3.0, -2.0, -2.0], [-2.0, 3.0, -2.0]],
                dtype=torch.float32,
            ),
            sampled_point_logits=torch.tensor(
                [
                    [[2.5, -1.0, -1.0], [2.0, -0.5, -1.0]],
                    [[-0.5, 0.5, 0.0], [-0.5, 0.5, 0.0]],
                ],
                dtype=torch.float32,
            ),
            sampled_point_indices=sampled_point_indices,
            sampled_point_mask=torch.ones(2, 2, dtype=torch.bool),
            slot_logits=torch.tensor(
                [
                    [[3.2, -1.5, -1.5], [1.9, -0.4, -1.0]],
                    [[-1.5, 2.7, -1.5], [-0.4, 0.6, -0.2]],
                ],
                dtype=torch.float32,
            ),
            slot_masses=torch.tensor([[0.8, 0.2], [0.6, 0.4]], dtype=torch.float32),
        )
        gt_labels = torch.tensor([0, 0, 1, 1], dtype=torch.long)

        weighted_total_loss, _ = compute_bsr_losses(
            bsr_output=output,
            gt_labels=gt_labels,
            num_classes=3,
            ignore_index=3,
            lambda_refine=0.0,
            lambda_consistency=1.0,
            score_weighting=True,
        )
        unweighted_total_loss, _ = compute_bsr_losses(
            bsr_output=output,
            gt_labels=gt_labels,
            num_classes=3,
            ignore_index=3,
            lambda_refine=0.0,
            lambda_consistency=1.0,
            score_weighting=False,
        )

        assert weighted_total_loss.ndim == 0
        assert unweighted_total_loss.ndim == 0
        assert not torch.isclose(weighted_total_loss, unweighted_total_loss)

    def test_consistency_uses_fused_candidate_logits_when_available(self):
        candidate_indices = torch.tensor([0], dtype=torch.long)
        output = BSROutput(
            candidate_indices=candidate_indices,
            candidate_scores=torch.tensor([1.0], dtype=torch.float32),
            refined_sp_features=torch.randn(1, 4),
            refined_sp_logits=torch.tensor([[4.0, -4.0, -4.0]], dtype=torch.float32),
            sampled_point_logits=torch.tensor([[[0.0, 0.0, 0.0]]], dtype=torch.float32),
            fused_candidate_logits=torch.tensor([[0.2, 0.1, -0.1]], dtype=torch.float32),
            sampled_point_indices=torch.tensor([[0]], dtype=torch.long),
            sampled_point_mask=torch.ones(1, 1, dtype=torch.bool),
            slot_logits=torch.tensor([[[4.0, -4.0, -4.0], [-4.0, 4.0, -4.0]]], dtype=torch.float32),
            slot_masses=torch.tensor([[0.5, 0.5]], dtype=torch.float32),
        )

        total_loss, loss_terms = compute_bsr_losses(
            bsr_output=output,
            gt_labels=torch.tensor([0], dtype=torch.long),
            num_classes=3,
            ignore_index=3,
            lambda_refine=0.0,
            lambda_consistency=1.0,
        )

        assert total_loss.item() > 0.0
        assert loss_terms["consistency_loss"].item() > 0.0
