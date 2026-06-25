import os
import sys

import pytest
import torch


project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, "src"))

from src.manuscript_tools.semantic_boundary import (  # noqa: E402
    aggregate_semantic_boundary_stats,
    compute_semantic_boundary_scene_stats,
)


def test_semantic_boundary_metrics_are_perfect_for_identical_predictions():
    coords = torch.tensor(
        [
            [0.00, 0.0, 0.0],
            [0.05, 0.0, 0.0],
            [0.10, 0.0, 0.0],
            [0.14, 0.0, 0.0],
            [0.20, 0.0, 0.0],
            [0.25, 0.0, 0.0],
        ],
        dtype=torch.float32,
    )
    gt = torch.tensor([0, 0, 0, 1, 1, 1], dtype=torch.long)

    stats = compute_semantic_boundary_scene_stats(
        point_coords=coords,
        gt_semantic=gt,
        pred_semantic=gt.clone(),
        num_classes=2,
        boundary_distance=0.08,
    )
    summary = aggregate_semantic_boundary_stats([stats])

    assert summary["overall_miou"] == pytest.approx(100.0)
    assert summary["transition_miou"] == pytest.approx(100.0)
    assert summary["boundary_iou"] == pytest.approx(1.0)


def test_semantic_boundary_metrics_drop_when_boundary_prediction_drifts():
    coords = torch.tensor(
        [
            [0.00, 0.0, 0.0],
            [0.05, 0.0, 0.0],
            [0.10, 0.0, 0.0],
            [0.14, 0.0, 0.0],
            [0.20, 0.0, 0.0],
            [0.25, 0.0, 0.0],
        ],
        dtype=torch.float32,
    )
    gt = torch.tensor([0, 0, 0, 1, 1, 1], dtype=torch.long)
    pred = torch.tensor([0, 0, 1, 1, 1, 1], dtype=torch.long)

    stats = compute_semantic_boundary_scene_stats(
        point_coords=coords,
        gt_semantic=gt,
        pred_semantic=pred,
        num_classes=2,
        boundary_distance=0.08,
    )
    summary = aggregate_semantic_boundary_stats([stats])

    assert summary["overall_miou"] < 100.0
    assert summary["transition_miou"] < 100.0
    assert summary["boundary_iou"] < 1.0
