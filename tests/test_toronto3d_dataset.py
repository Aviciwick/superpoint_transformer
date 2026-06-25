import os
import sys

import numpy as np
from omegaconf import OmegaConf
from plyfile import PlyData, PlyElement
import torch


project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, "src"))

from src.datasets.toronto3d import read_toronto3d_cloud
from src.datasets.toronto3d_config import (
    CLASS_NAMES,
    ID2TRAINID,
    TORONTO3D_NUM_CLASSES,
    TORONTO3D_SPLITS,
)


def test_toronto3d_default_protocol_uses_whole_roads_and_l002_test():
    cfg = OmegaConf.load(os.path.join(project_root, "configs/datamodule/semantic/toronto3d.yaml"))
    experiments = os.path.join(project_root, "configs/experiment/semantic")

    assert TORONTO3D_SPLITS == {
        "train": ["L001", "L003", "L004"],
        "val": [],
        "test": ["L002"],
    }
    assert cfg.trainval is False
    assert cfg.val_on_test is True
    assert cfg.xy_tiling is None
    assert cfg.pc_tiling is None
    assert "intensity" in cfg.point_hf
    assert cfg.custom_hash == "toronto3d_spt64_intensity_wholeroad_v1"
    assert not os.path.exists(os.path.join(experiments, "toronto3d_high_budget.yaml"))
    assert not os.path.exists(os.path.join(experiments, "toronto3d_bsr_sampled_point_only.yaml"))
    assert not os.path.exists(os.path.join(experiments, "toronto3d_bsr_point_refinement_control.yaml"))


def test_toronto3d_parser_maps_raw_labels_and_features(tmp_path):
    vertex = np.asarray(
        [
            (1.0, 2.0, 3.0, 10, 20, 30, 5.0, 0.0, 0.0, 0.0),
            (2.0, 3.0, 4.0, 40, 50, 60, 10.0, 0.0, 0.0, 1.0),
            (3.0, 4.0, 5.0, 70, 80, 90, 20.0, 0.0, 0.0, 8.0),
        ],
        dtype=[
            ("x", "f8"),
            ("y", "f8"),
            ("z", "f8"),
            ("red", "u1"),
            ("green", "u1"),
            ("blue", "u1"),
            ("scalar_Intensity", "f4"),
            ("scalar_GPSTime", "f4"),
            ("scalar_ScanAngleRank", "f4"),
            ("scalar_Label", "f4"),
        ],
    )
    path = tmp_path / "L999.ply"
    PlyData([PlyElement.describe(vertex, "vertex")], text=False).write(path)

    data = read_toronto3d_cloud(str(path))

    assert data.pos.shape == (3, 3)
    assert torch.allclose(data.pos[0], torch.zeros(3))
    assert data.rgb.shape == (3, 3)
    assert data.rgb.max() <= 1.0
    assert data.intensity.shape == (3, 1)
    assert torch.allclose(data.intensity[:, 0], torch.tensor([0.25, 0.5, 1.0]))
    assert data.y.tolist() == [int(ID2TRAINID[0]), int(ID2TRAINID[1]), int(ID2TRAINID[8])]
    assert CLASS_NAMES[-1] == "ignored"
    assert TORONTO3D_NUM_CLASSES == 8


def test_toronto3d_parser_sanitizes_nonfinite_intensity(tmp_path):
    vertex = np.asarray(
        [
            (1.0, 2.0, 3.0, 10, 20, 30, np.nan, 0.0),
            (2.0, 3.0, 4.0, 40, 50, 60, 10.0, 1.0),
            (3.0, 4.0, 5.0, 70, 80, 90, 20.0, 8.0),
        ],
        dtype=[
            ("x", "f8"),
            ("y", "f8"),
            ("z", "f8"),
            ("red", "u1"),
            ("green", "u1"),
            ("blue", "u1"),
            ("scalar_Intensity", "f4"),
            ("scalar_Label", "f4"),
        ],
    )
    path = tmp_path / "L998.ply"
    PlyData([PlyElement.describe(vertex, "vertex")], text=False).write(path)

    data = read_toronto3d_cloud(str(path))

    assert data.intensity.shape == (3, 1)
    assert torch.isfinite(data.intensity).all()
    assert torch.allclose(data.intensity[:, 0], torch.tensor([0.0, 0.5, 1.0]))
