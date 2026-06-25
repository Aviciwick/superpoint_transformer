import torch

from src.data import Data, NAG
from src.transforms.graph import (
    NAGAddSelfLoops,
    OnTheFlyHorizontalEdgeFeatures,
    RadiusHorizontalGraph,
)


def test_radius_horizontal_graph_allows_single_node_level():
    edge_keys = [
        "mean_off",
        "std_off",
        "mean_dist",
        "angle_source",
        "angle_target",
        "centroid_dir",
        "centroid_dist",
        "normal_angle",
        "log_length",
        "log_surface",
        "log_volume",
        "log_size",
    ]
    nag = NAG(
        [
            Data(
                pos=torch.randn(10, 3),
                super_index=torch.zeros(10, dtype=torch.long),
            ),
            Data(
                pos=torch.randn(1, 3),
                normal=torch.randn(1, 3),
                log_length=torch.zeros(1),
                log_surface=torch.zeros(1),
                log_volume=torch.zeros(1),
                log_size=torch.zeros(1),
            ),
        ]
    )

    nag = RadiusHorizontalGraph(
        k_min=1,
        k_max=4,
        gap=0.2,
        keys=["mean_off", "std_off", "mean_dist"],
    )(nag)
    assert nag[1].edge_index.shape == (2, 0)
    assert nag[1].edge_attr.shape == (0, 7)

    nag = OnTheFlyHorizontalEdgeFeatures(keys=edge_keys)(nag)
    assert nag[1].edge_index.shape == (2, 0)
    assert nag[1].edge_attr.shape == (0, 18)

    nag = NAGAddSelfLoops()(nag)
    assert nag[1].edge_index.shape == (2, 1)
    assert nag[1].edge_attr.shape == (1, 18)


def test_radius_horizontal_graph_connects_when_radius_search_is_empty():
    nag = NAG(
        [
            Data(
                pos=torch.tensor([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]]),
                super_index=torch.tensor([0, 1]),
            ),
            Data(pos=torch.tensor([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]])),
        ]
    )

    nag = RadiusHorizontalGraph(
        k_min=1,
        k_max=1,
        gap=0.0,
        se_min=1,
        keys=["mean_off", "std_off", "mean_dist"],
    )(nag)

    assert nag[1].edge_index.shape == (2, 1)
    assert nag[1].edge_attr.shape == (1, 7)
