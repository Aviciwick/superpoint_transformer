import os
import sys

import numpy as np
from omegaconf import OmegaConf
from plyfile import PlyData, PlyElement
import pytest
import torch


project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, "src"))

from src.datasets.sensaturban import (
    build_sensaturban_adaptive_tiling_manifest,
    compute_adaptive_xy_tiling,
    discover_sensaturban_splits,
    expand_sensaturban_adaptive_cloud_ids,
    resolve_sensaturban_splits,
    read_sensaturban_cloud,
)
from src.datasets.base import BaseDataset
from src.datasets.sensaturban_config import (
    CLASS_NAMES,
    SENSATURBAN_NUM_CLASSES,
    STUFF_CLASSES,
)


def _write_ply(path, vertex):
    PlyData([PlyElement.describe(vertex, "vertex")], text=False).write(path)


def test_sensaturban_parser_reads_labelled_cloud(tmp_path):
    vertex = np.asarray(
        [
            (10.0, 20.0, 30.0, 10, 20, 30, 0),
            (11.0, 22.0, 33.0, 40, 50, 60, 12),
            (12.0, 24.0, 36.0, 70, 80, 90, 5),
        ],
        dtype=[
            ("x", "f4"),
            ("y", "f4"),
            ("z", "f4"),
            ("red", "u1"),
            ("green", "u1"),
            ("blue", "u1"),
            ("class", "u1"),
        ],
    )
    path = tmp_path / "block.ply"
    _write_ply(path, vertex)

    data = read_sensaturban_cloud(str(path))

    assert data.pos.shape == (3, 3)
    assert torch.allclose(data.pos[0], torch.zeros(3))
    assert torch.allclose(data.pos_offset, torch.tensor([10.0, 20.0, 30.0]))
    assert data.rgb.shape == (3, 3)
    assert data.rgb.max() <= 1.0
    assert data.y.tolist() == [0, 12, 5]
    assert getattr(data, "intensity", None) is None
    assert getattr(data, "obj", None) is None


def test_sensaturban_parser_allows_unlabelled_cloud(tmp_path):
    vertex = np.asarray(
        [
            (1.0, 2.0, 3.0, 10, 20, 30),
            (2.0, 3.0, 4.0, 40, 50, 60),
        ],
        dtype=[
            ("x", "f4"),
            ("y", "f4"),
            ("z", "f4"),
            ("red", "u1"),
            ("green", "u1"),
            ("blue", "u1"),
        ],
    )
    path = tmp_path / "test_block.ply"
    _write_ply(path, vertex)

    data = read_sensaturban_cloud(str(path), semantic=False)

    assert data.pos.shape == (2, 3)
    assert data.rgb.shape == (2, 3)
    assert getattr(data, "y", None) is None


def test_sensaturban_parser_can_tolerate_unlabelled_test_cloud(tmp_path):
    vertex = np.asarray(
        [(1.0, 2.0, 3.0, 10, 20, 30)],
        dtype=[
            ("x", "f4"),
            ("y", "f4"),
            ("z", "f4"),
            ("red", "u1"),
            ("green", "u1"),
            ("blue", "u1"),
        ],
    )
    path = tmp_path / "unlabelled_test_block.ply"
    _write_ply(path, vertex)

    data = read_sensaturban_cloud(str(path), semantic=True, allow_unlabelled=True)

    assert data.pos.shape == (1, 3)
    assert getattr(data, "y", None) is None


def test_sensaturban_parser_rejects_truncated_ply(tmp_path):
    path = tmp_path / "broken_block.ply"
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        "element vertex 3\n"
        "property float32 x\n"
        "property float32 y\n"
        "property float32 z\n"
        "property uint8 red\n"
        "property uint8 green\n"
        "property uint8 blue\n"
        "property uint8 class\n"
        "end_header\n"
    ).encode("ascii")
    one_record = np.asarray(
        [(1.0, 2.0, 3.0, 10, 20, 30, 1)],
        dtype=[
            ("x", "f4"),
            ("y", "f4"),
            ("z", "f4"),
            ("red", "u1"),
            ("green", "u1"),
            ("blue", "u1"),
            ("class", "u1"),
        ],
    ).tobytes()
    path.write_bytes(header + one_record)

    with pytest.raises(RuntimeError, match="SensatUrban PLY size mismatch.*broken_block"):
        read_sensaturban_cloud(str(path))


def test_sensaturban_config_and_dynamic_split_discovery(tmp_path):
    (tmp_path / "SensatUrban" / "train").mkdir(parents=True)
    (tmp_path / "SensatUrban" / "val").mkdir()
    (tmp_path / "SensatUrban" / "test").mkdir()
    for rel in (
            "train/cambridge_block_2.ply",
            "train/birmingham_block_4.ply",
            "val/cambridge_block_12.ply",
            "test/birmingham_block_2.ply"):
        (tmp_path / "SensatUrban" / rel).write_bytes(b"")

    splits = discover_sensaturban_splits(str(tmp_path / "SensatUrban"))
    cfg = OmegaConf.load(os.path.join(project_root, "configs/datamodule/semantic/sensaturban.yaml"))

    assert splits == {
        "train": ["birmingham_block_4", "cambridge_block_2"],
        "val": ["cambridge_block_12"],
        "test": ["birmingham_block_2"],
    }
    assert SENSATURBAN_NUM_CLASSES == 13
    assert CLASS_NAMES[:3] == ["ground", "vegetation", "building"]
    assert CLASS_NAMES[-1] == "ignored"
    assert STUFF_CLASSES == list(range(13))
    assert cfg.num_classes == 13
    assert cfg.instance is False
    assert cfg.xy_tiling is None
    assert cfg.pc_tiling is None
    assert cfg.adaptive_xy_tiling.enable is True
    assert cfg.adaptive_xy_tiling.reference_cloud == "birmingham_block_9"
    assert cfg.adaptive_xy_tiling.reference_xy_tiling == 3
    assert cfg.adaptive_xy_tiling.min_tile_points == 1024
    assert cfg.voxel == 0.2
    assert cfg.custom_hash == "sensaturban_spt64_rgb_adaptxy_refb9x3_vox020_cityfixed_pruned_v1"
    assert "rgb" in cfg.partition_hf
    assert "intensity" not in cfg.point_hf


def test_sensaturban_adaptive_xy_tiling_uses_reference_tile_scale():
    assert compute_adaptive_xy_tiling(
        width=900.0,
        height=900.0,
        tile_size_xy=(300.0, 300.0),
        max_tiles_per_axis=12,
    ) == (3, 3)
    assert compute_adaptive_xy_tiling(
        width=1200.0,
        height=300.0,
        tile_size_xy=(300.0, 300.0),
        max_tiles_per_axis=12,
    ) == (4, 1)
    assert compute_adaptive_xy_tiling(
        width=100.0,
        height=100.0,
        tile_size_xy=(300.0, 300.0),
        max_tiles_per_axis=12,
    ) == (1, 1)


def test_sensaturban_adaptive_manifest_and_cloud_id_expansion(tmp_path):
    root = tmp_path / "SensatUrban"
    root.mkdir()
    reference_rows = []
    for x in (150.0, 450.0, 750.0):
        for y in (150.0, 450.0, 750.0):
            reference_rows.append((x, y, 0.0, 10, 20, 30, 0))
    reference = np.asarray(
        reference_rows + [(0.0, 0.0, 0.0, 10, 20, 30, 0), (900.0, 900.0, 0.0, 10, 20, 30, 1)],
        dtype=[
            ("x", "f4"),
            ("y", "f4"),
            ("z", "f4"),
            ("red", "u1"),
            ("green", "u1"),
            ("blue", "u1"),
            ("class", "u1"),
        ],
    )
    long_rows = [(150.0 + 300.0 * i, 150.0, 0.0, 10, 20, 30, 0) for i in range(4)]
    long_block = np.asarray(
        long_rows + [(0.0, 0.0, 0.0, 10, 20, 30, 0), (1200.0, 300.0, 0.0, 10, 20, 30, 1)],
        dtype=reference.dtype,
    )
    _write_ply(root / "birmingham_block_9.ply", reference)
    _write_ply(root / "cambridge_block_2.ply", long_block)

    config = {
        "enable": True,
        "reference_cloud": "birmingham_block_9",
        "reference_xy_tiling": 3,
        "max_tiles_per_axis": 12,
        "min_tile_points": 1,
        "chunk_size": 4,
        "manifest_name": "adaptive_test_manifest.json",
    }
    manifest = build_sensaturban_adaptive_tiling_manifest(
        str(root),
        {
            "train": ["birmingham_block_9"],
            "val": ["cambridge_block_2"],
            "test": [],
        },
        config,
    )
    expanded = expand_sensaturban_adaptive_cloud_ids(
        {
            "train": ["birmingham_block_9"],
            "val": ["cambridge_block_2"],
            "test": [],
        },
        manifest,
    )

    assert manifest["clouds"]["birmingham_block_9"]["tiling"] == [3, 3]
    assert manifest["clouds"]["birmingham_block_9"]["tile_count"] == 9
    assert manifest["clouds"]["cambridge_block_2"]["tiling"] == [4, 1]
    assert manifest["clouds"]["cambridge_block_2"]["tile_count"] == 4
    assert len(expanded["train"]) == 9
    assert expanded["train"][0] == "birmingham_block_9__TILE_1-1_OF_3-3"
    assert expanded["train"][-1] == "birmingham_block_9__TILE_3-3_OF_3-3"
    assert expanded["val"] == [
        "cambridge_block_2__TILE_1-1_OF_4-1",
        "cambridge_block_2__TILE_2-1_OF_4-1",
        "cambridge_block_2__TILE_3-1_OF_4-1",
        "cambridge_block_2__TILE_4-1_OF_4-1",
    ]


def test_sensaturban_adaptive_manifest_skips_empty_grid_tiles(tmp_path):
    root = tmp_path / "SensatUrban"
    root.mkdir()
    vertex = np.asarray(
        [
            (0.0, 0.0, 0.0, 10, 20, 30, 0),
            (250.0, 250.0, 0.0, 10, 20, 30, 1),
            (750.0, 250.0, 0.0, 10, 20, 30, 2),
            (250.0, 750.0, 0.0, 10, 20, 30, 3),
            (1000.0, 1000.0, 0.0, 10, 20, 30, 4),
        ],
        dtype=[
            ("x", "f4"),
            ("y", "f4"),
            ("z", "f4"),
            ("red", "u1"),
            ("green", "u1"),
            ("blue", "u1"),
            ("class", "u1"),
        ],
    )
    _write_ply(root / "birmingham_block_9.ply", vertex)

    config = {
        "enable": True,
        "reference_cloud": "birmingham_block_9",
        "reference_xy_tiling": 2,
        "max_tiles_per_axis": 12,
        "min_tile_points": 1,
        "chunk_size": 4,
        "manifest_name": "adaptive_empty_test_manifest.json",
    }
    splits = {"train": ["birmingham_block_9"], "val": [], "test": []}
    manifest = build_sensaturban_adaptive_tiling_manifest(str(root), splits, config)
    expanded = expand_sensaturban_adaptive_cloud_ids(splits, manifest)

    assert manifest["clouds"]["birmingham_block_9"]["tiling"] == [2, 2]
    assert manifest["clouds"]["birmingham_block_9"]["occupied_tiles"] == [[1, 1], [1, 2], [2, 1]]
    assert manifest["clouds"]["birmingham_block_9"]["tile_count"] == 3
    assert expanded["train"] == [
        "birmingham_block_9__TILE_1-1_OF_2-2",
        "birmingham_block_9__TILE_1-2_OF_2-2",
        "birmingham_block_9__TILE_2-1_OF_2-2",
    ]


def test_sensaturban_adaptive_manifest_filters_sparse_grid_tiles(tmp_path):
    root = tmp_path / "SensatUrban"
    root.mkdir()
    dense_tile_points = [(250.0 + i, 250.0, 0.0, 10, 20, 30, 0) for i in range(4)]
    sparse_tile_points = [(750.0, 750.0, 0.0, 10, 20, 30, 1) for _ in range(3)]
    vertex = np.asarray(
        dense_tile_points + sparse_tile_points + [(0.0, 0.0, 0.0, 10, 20, 30, 0), (1000.0, 1000.0, 0.0, 10, 20, 30, 1)],
        dtype=[
            ("x", "f4"),
            ("y", "f4"),
            ("z", "f4"),
            ("red", "u1"),
            ("green", "u1"),
            ("blue", "u1"),
            ("class", "u1"),
        ],
    )
    _write_ply(root / "birmingham_block_9.ply", vertex)

    config = {
        "enable": True,
        "reference_cloud": "birmingham_block_9",
        "reference_xy_tiling": 2,
        "max_tiles_per_axis": 12,
        "min_tile_points": 4,
        "chunk_size": 4,
        "manifest_name": "adaptive_sparse_test_manifest.json",
    }
    splits = {"train": ["birmingham_block_9"], "val": [], "test": []}
    manifest = build_sensaturban_adaptive_tiling_manifest(str(root), splits, config)
    expanded = expand_sensaturban_adaptive_cloud_ids(splits, manifest)

    assert manifest["version"] == 3
    assert manifest["clouds"]["birmingham_block_9"]["tile_point_counts"] == [
        [1, 1, 5],
        [2, 2, 3],
    ]
    assert manifest["clouds"]["birmingham_block_9"]["occupied_tiles"] == [[1, 1]]
    assert expanded["train"] == ["birmingham_block_9__TILE_1-1_OF_2-2"]


def test_base_dataset_strips_tile_suffix_without_global_tiling():
    dataset = BaseDataset.__new__(BaseDataset)
    dataset.xy_tiling = None
    dataset.pc_tiling = None

    assert dataset.id_to_base_id("cambridge_block_2__TILE_3-1_OF_4-1") == "cambridge_block_2"


def test_sensaturban_resolves_missing_val_from_train_folder(tmp_path):
    root = tmp_path / "SensatUrban"
    (root / "train").mkdir(parents=True)
    (root / "test").mkdir()
    for name in (
            "birmingham_block_0.ply",
            "birmingham_block_1.ply",
            "cambridge_block_0.ply",
            "cambridge_block_1.ply",
            "cambridge_block_2.ply"):
        (root / "train" / name).write_bytes(b"")
    (root / "test" / "birmingham_block_2.ply").write_bytes(b"")

    splits = resolve_sensaturban_splits(
        str(root),
        val_from_train=True,
        val_count=2,
    )

    assert len(splits["val"]) == 2
    assert set(splits["val"]).issubset({
        "birmingham_block_0",
        "birmingham_block_1",
        "cambridge_block_0",
        "cambridge_block_1",
        "cambridge_block_2",
    })
    assert set(splits["train"]).isdisjoint(splits["val"])
    assert len(splits["train"]) == 3
    assert splits["test"] == ["birmingham_block_2"]


def test_sensaturban_default_protocol_uses_city_fixed_flat_split(tmp_path):
    root = tmp_path / "SensatUrban"
    root.mkdir()
    for idx in range(14):
        (root / f"birmingham_block_{idx}.ply").write_bytes(b"")
    for idx in (
            2, 3, 4, 6, 7, 8, 9, 10,
            12, 13, 14, 15, 16, 17, 18, 19, 20, 21,
            22, 23, 25, 26, 27, 28, 32, 33):
        (root / f"cambridge_block_{idx}.ply").write_bytes(b"")

    splits = resolve_sensaturban_splits(
        str(root),
        split_protocol="city_fixed",
    )
    cfg = OmegaConf.load(os.path.join(project_root, "configs/datamodule/semantic/sensaturban.yaml"))

    assert splits["train"] == [
        "birmingham_block_0",
        "birmingham_block_1",
        "birmingham_block_2",
        "birmingham_block_3",
        "birmingham_block_4",
        "birmingham_block_5",
        "birmingham_block_6",
        "birmingham_block_7",
        "birmingham_block_8",
        "birmingham_block_9",
        "cambridge_block_2",
        "cambridge_block_3",
        "cambridge_block_4",
        "cambridge_block_6",
        "cambridge_block_7",
        "cambridge_block_8",
        "cambridge_block_9",
        "cambridge_block_10",
        "cambridge_block_12",
        "cambridge_block_13",
        "cambridge_block_14",
        "cambridge_block_15",
        "cambridge_block_16",
        "cambridge_block_17",
        "cambridge_block_18",
        "cambridge_block_19",
        "cambridge_block_20",
        "cambridge_block_21",
    ]
    assert splits["val"] == [
        "birmingham_block_10",
        "birmingham_block_11",
        "cambridge_block_22",
        "cambridge_block_23",
        "cambridge_block_25",
        "cambridge_block_26",
    ]
    assert splits["test"] == [
        "birmingham_block_12",
        "birmingham_block_13",
        "cambridge_block_27",
        "cambridge_block_28",
        "cambridge_block_32",
        "cambridge_block_33",
    ]
    assert cfg.split_protocol == "city_fixed"
    assert cfg.val_from_train is False


def test_sensaturban_respects_explicit_splits_and_excludes(tmp_path):
    root = tmp_path / "SensatUrban"
    (root / "train").mkdir(parents=True)
    (root / "val").mkdir()
    (root / "test").mkdir()
    for rel in (
            "train/a.ply",
            "train/b.ply",
            "train/broken.ply",
            "val/c.ply",
            "test/d.ply"):
        (root / rel).write_bytes(b"")

    splits = resolve_sensaturban_splits(
        str(root),
        train_clouds=["a", "broken"],
        val_clouds=["b"],
        test_clouds=["d"],
        exclude_clouds=["broken"],
        val_from_train=True,
    )

    assert splits == {
        "train": ["a"],
        "val": ["b"],
        "test": ["d"],
    }


def test_sensaturban_explicit_splits_override_city_fixed_counts(tmp_path):
    root = tmp_path / "SensatUrban"
    root.mkdir()
    for name in ("a.ply", "b.ply", "c.ply"):
        (root / name).write_bytes(b"")

    splits = resolve_sensaturban_splits(
        str(root),
        split_protocol="city_fixed",
        train_clouds=["a"],
        val_clouds=["b"],
        test_clouds=["c"],
    )

    assert splits == {
        "train": ["a"],
        "val": ["b"],
        "test": ["c"],
    }
