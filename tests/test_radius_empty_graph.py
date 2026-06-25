import torch

from src.utils.neighbors import cluster_radius_nn_graph
from src.utils.scatter import scatter_nearest_neighbor


def test_scatter_nearest_neighbor_accepts_empty_edge_index():
    points = torch.randn(4, 3)
    index = torch.tensor([0, 0, 1, 1])
    edge_index = torch.empty((2, 0), dtype=torch.long)

    candidate, candidate_idx = scatter_nearest_neighbor(
        points,
        index,
        edge_index,
        chunk_size=10,
    )

    assert candidate.shape == (0, 3)
    assert candidate_idx.shape == (2, 0)


def test_cluster_radius_nn_graph_returns_empty_when_no_candidate_edges():
    points = torch.tensor([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]])
    index = torch.tensor([0, 1])

    edge_index, distances = cluster_radius_nn_graph(
        points,
        index,
        k_max=1,
        gap=0.0,
        chunk_size=10,
    )

    assert edge_index.shape == (2, 0)
    assert distances.shape == (0,)
