import torch

from src.data import Data
from src.transforms.sampling import (
    GridSampling3D,
    SampleRecursiveMainXYAxisTiling,
)


def test_grid_sampling_reports_empty_cloud_clearly():
    data = Data(pos=torch.empty((0, 3)))

    try:
        GridSampling3D(size=1.0)(data)
    except ValueError as exc:
        assert "empty point cloud" in str(exc)
    else:
        raise AssertionError("GridSampling3D should reject empty point clouds")


def test_grid_sampling_chunking_preserves_single_pos_offset():
    data = Data(
        pos=torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [0.1, 0.0, 0.0],
                [2.0, 0.0, 0.0],
                [2.1, 0.0, 0.0],
                [4.0, 0.0, 0.0],
                [4.1, 0.0, 0.0],
            ],
            dtype=torch.float32,
        ),
        pos_offset=torch.tensor([10.0, 20.0, 30.0], dtype=torch.float64),
        y=torch.tensor([0, 0, 1, 1, 2, 2], dtype=torch.long),
    )

    sampled = GridSampling3D(
        size=1.0,
        hist_key="y",
        hist_size=3,
        inplace=False,
        chunk_size=2,
    )(data)

    assert sampled.pos_offset.shape == (3,)
    assert torch.equal(sampled.pos_offset, data.pos_offset)


def test_recursive_pc_tiling_avoids_empty_split_for_degenerate_xy():
    data = Data(
        pos=torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0],
                [0.0, 0.0, 2.0],
                [0.0, 0.0, 3.0],
            ]
        )
    )

    left, right = SampleRecursiveMainXYAxisTiling.split_by_main_xy_direction(data)

    assert left.num_points == 2
    assert right.num_points == 2
