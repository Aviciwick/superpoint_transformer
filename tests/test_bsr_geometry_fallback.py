import os
import sys

import torch


project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, "src"))

from src.bsr.geometry import extract_selector_handcrafted_features


class DummyLevel:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)


def test_extract_handcrafted_features_falls_back_to_level0_and_log_geometry():
    geo_feature_keys = [
        "linearity",
        "planarity",
        "scattering",
        "verticality",
        "curvature",
        "length",
        "surface",
        "volume",
    ]

    nag = [
        DummyLevel(
            super_index=torch.tensor([0, 0, 1, 1], dtype=torch.long),
            linearity=torch.tensor([[1.0], [3.0], [5.0], [7.0]]),
            planarity=torch.tensor([[2.0], [4.0], [6.0], [8.0]]),
            scattering=torch.tensor([[0.2], [0.6], [0.4], [0.8]]),
            verticality=torch.tensor([[0.1], [0.3], [0.2], [0.4]]),
            curvature=torch.tensor([[0.5], [0.7], [0.9], [1.1]]),
        ),
        DummyLevel(
            num_nodes=2,
            pos=torch.zeros(2, 3),
            log_length=torch.tensor([[1.5], [2.5]]),
            log_surface=torch.tensor([[0.4], [0.8]]),
            log_volume=torch.tensor([[0.2], [0.9]]),
        ),
    ]

    features, missing_keys = extract_selector_handcrafted_features(
        nag=nag,
        preferred_keys=geo_feature_keys,
        device=nag[1].pos.device,
    )

    expected = torch.tensor(
        [
            [2.0, 3.0, 0.4, 0.2, 0.6, 1.5, 0.4, 0.2],
            [6.0, 7.0, 0.6, 0.3, 1.0, 2.5, 0.8, 0.9],
        ],
        dtype=torch.float32,
    )

    assert features.shape == (2, 8)
    assert torch.allclose(features, expected)
    assert missing_keys == []
