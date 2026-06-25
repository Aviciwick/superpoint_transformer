import os
import sys

import torch


project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, "src"))

from src.bsr import (  # noqa: E402
    aggregate_mixed_superpoint_records,
    build_mixed_superpoint_scene_records,
)


def test_mixed_superpoint_diagnostics_highlight_mixed_bucket():
    coords = torch.tensor(
        [
            [0.00, 0.0, 0.0],
            [0.04, 0.0, 0.0],
            [0.08, 0.0, 0.0],
            [0.12, 0.0, 0.0],
            [0.16, 0.0, 0.0],
            [0.20, 0.0, 0.0],
        ],
        dtype=torch.float32,
    )
    gt = torch.tensor([0, 0, 0, 1, 1, 1], dtype=torch.long)
    baseline_pred = torch.tensor([0, 0, 0, 0, 1, 1], dtype=torch.long)
    bsr_pred = torch.tensor([0, 0, 0, 1, 1, 1], dtype=torch.long)
    super_index = torch.tensor([0, 0, 1, 1, 2, 2], dtype=torch.long)

    scene = build_mixed_superpoint_scene_records(
        point_coords=coords,
        gt_labels=gt,
        baseline_pred=baseline_pred,
        bsr_pred=bsr_pred,
        super_index=super_index,
        num_classes=2,
        boundary_distance=0.08,
        candidate_mask=torch.tensor([0, 1, 0], dtype=torch.bool),
        assignment_entropy=torch.tensor([0.0, 0.6, 0.0], dtype=torch.float32),
        secondary_slot_mass=torch.tensor([0.0, 0.35, 0.0], dtype=torch.float32),
        slot_diversity=torch.tensor([0.0, 0.4, 0.0], dtype=torch.float32),
    )
    summary = aggregate_mixed_superpoint_records([scene])

    purity = {bucket["bucket"]: bucket for bucket in summary["purity_buckets"]}
    transition = {bucket["bucket"]: bucket for bucket in summary["transition_buckets"]}
    slices = {bucket["bucket"]: bucket for bucket in summary["diagnostic_slices"]}

    assert purity["heavily_mixed"]["superpoints"] == 1
    assert purity["heavily_mixed"]["candidate_hit_rate"] == 1.0
    assert purity["heavily_mixed"]["bsr_point_acc_gain"] > 0.0
    assert purity["heavily_mixed"]["secondary_slot_mass"] > purity["clean"]["secondary_slot_mass"]
    assert transition["transition_heavy"]["secondary_slot_mass"] >= transition["non_transition"]["secondary_slot_mass"]
    assert summary["diagnostic_slice_rules"]["mutually_exclusive"]
    assert set(slices) == {"clean", "transition_heavy", "mixed_content"}
    assert slices["mixed_content"]["superpoints"] == purity["heavily_mixed"]["superpoints"]
    assert sum(bucket["superpoints"] for bucket in slices.values()) <= summary["num_superpoints"]
