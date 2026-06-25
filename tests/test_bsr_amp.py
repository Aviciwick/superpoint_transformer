import os
import sys

import torch
import torch.nn.functional as F


project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, "src"))

from src.bsr import BSRModule, BoundaryPriorSelector


def test_selector_boundary_score_accepts_mixed_precision_inputs():
    selector = BoundaryPriorSelector(
        topk_ratio=0.5,
        score_terms=("boundary", "geometry", "uncertainty"),
        term_weights={"boundary": 1.0, "geometry": 1.0, "uncertainty": 1.0},
    )
    coarse_logits = torch.tensor(
        [
            [5.0, -5.0, 0.0],
            [-4.0, 4.0, 0.0],
            [1.0, 0.5, -1.5],
            [0.1, -0.2, 0.3],
        ],
        dtype=torch.float16,
    )
    handcrafted_features = torch.randn(4, 5, dtype=torch.float32)
    edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 3]], dtype=torch.long)

    candidate_indices, candidate_scores = selector(
        coarse_logits=coarse_logits,
        handcrafted_features=handcrafted_features,
        edge_index=edge_index,
    )

    assert candidate_indices.numel() == 2
    assert candidate_scores.dtype == torch.float32
    assert torch.isfinite(candidate_scores).all()


def test_bsr_module_forward_handles_half_logits_and_float_features():
    torch.manual_seed(7)
    module = BSRModule(
        d_model=16,
        num_classes=4,
        selector_topk_ratio=0.5,
        selector_score_terms=["uncertainty", "geometry", "boundary"],
        n_sample=4,
        d_raw=6,
        n_heads=4,
        variant="mixture",
        n_subregions=2,
    )

    sp_features = torch.randn(4, 16, dtype=torch.float32)
    sp_centroids = torch.randn(4, 3, dtype=torch.float32)
    coarse_logits = torch.randn(4, 4, dtype=torch.float16)
    packed_raw_points = torch.randn(4, 4, 6, dtype=torch.float32)
    packed_point_indices = torch.randint(0, 16, (4, 4))
    packed_point_mask = torch.ones(4, 4, dtype=torch.bool)
    handcrafted_features = torch.randn(4, 6, dtype=torch.float32)
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long)

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

    assert output.candidate_indices.numel() == 2
    assert output.candidate_scores.dtype == torch.float32
    assert torch.isfinite(output.candidate_scores).all()
    assert output.refined_sp_logits.dtype == coarse_logits.dtype


def test_bsr_losses_accept_mixed_precision_logits():
    from src.bsr import BSROutput, compute_bsr_losses

    output = BSROutput(
        candidate_indices=torch.tensor([0, 2], dtype=torch.long),
        candidate_scores=torch.tensor([0.8, 0.6], dtype=torch.float32),
        refined_sp_features=torch.randn(4, 8, dtype=torch.float32),
        refined_sp_logits=torch.randn(4, 3, dtype=torch.float16),
        sampled_point_logits=torch.randn(2, 3, 3, dtype=torch.float32),
        fused_candidate_logits=torch.randn(2, 3, dtype=torch.float16),
        sampled_point_indices=torch.tensor([[0, 1, 2], [3, 4, 5]], dtype=torch.long),
        sampled_point_mask=torch.ones(2, 3, dtype=torch.bool),
        boundary_logits=torch.randn(2, 3, dtype=torch.float32),
        boundary_scores=torch.full((2, 3), 0.5, dtype=torch.float32),
        slot_logits=torch.randn(2, 2, 3, dtype=torch.float32),
        slot_masses=torch.tensor([[0.6, 0.4], [0.55, 0.45]], dtype=torch.float32),
    )
    gt_labels = torch.tensor([0, 1, 2, 1, 0, 2], dtype=torch.long)

    total_loss, loss_terms = compute_bsr_losses(
        bsr_output=output,
        gt_labels=gt_labels,
        num_classes=3,
        ignore_index=3,
        lambda_refine=0.5,
        lambda_consistency=0.1,
        lambda_diversity=0.2,
        lambda_boundary=0.3,
    )

    assert total_loss.dtype == torch.float32
    assert torch.isfinite(total_loss)
    assert torch.isfinite(loss_terms["consistency_loss"])
    assert torch.isfinite(loss_terms["boundary_loss"])
