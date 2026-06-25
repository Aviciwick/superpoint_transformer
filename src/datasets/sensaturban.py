import json
import math
import os
import os.path as osp
import re
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from plyfile import PlyData

from src.data import Data
from src.datasets import BaseDataset
from src.datasets.sensaturban_config import *
from src.utils.color import to_float_rgb


__all__ = [
    "SensatUrban",
    "MiniSensatUrban",
    "build_sensaturban_adaptive_tiling_manifest",
    "compute_adaptive_xy_tiling",
    "discover_sensaturban_splits",
    "expand_sensaturban_adaptive_cloud_ids",
    "resolve_sensaturban_splits",
    "read_sensaturban_cloud",
]


# Match the DALES/KITTI workaround for large outdoor PLY datasets and
# multi-worker dataloading on some Linux setups.
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy("file_system")


_PLY_TYPE_BYTES = {
    "char": 1,
    "uchar": 1,
    "int8": 1,
    "uint8": 1,
    "short": 2,
    "ushort": 2,
    "int16": 2,
    "uint16": 2,
    "int": 4,
    "uint": 4,
    "int32": 4,
    "uint32": 4,
    "float": 4,
    "float32": 4,
    "double": 8,
    "float64": 8,
}

_PLY_NUMPY_DTYPES = {
    "char": "i1",
    "uchar": "u1",
    "int8": "i1",
    "uint8": "u1",
    "short": "<i2",
    "ushort": "<u2",
    "int16": "<i2",
    "uint16": "<u2",
    "int": "<i4",
    "uint": "<u4",
    "int32": "<i4",
    "uint32": "<u4",
    "float": "<f4",
    "float32": "<f4",
    "double": "<f8",
    "float64": "<f8",
}


def discover_sensaturban_splits(root: str) -> Dict[str, List[str]]:
    """Return sorted cloud ids from `train`, `val`, and `test` folders."""
    splits = {}
    for stage in ("train", "val", "test"):
        stage_dir = osp.join(root, stage)
        if not osp.isdir(stage_dir):
            splits[stage] = []
            continue
        splits[stage] = sorted(
            osp.splitext(name)[0]
            for name in os.listdir(stage_dir)
            if name.lower().endswith(".ply")
        )
    return splits


_BLOCK_ID_RE = re.compile(r"^(?P<city>birmingham|cambridge)_block_(?P<block>\d+)$")


def _discover_flat_cloud_ids(root: str) -> List[str]:
    if not osp.isdir(root):
        return []
    return sorted(
        osp.splitext(name)[0]
        for name in os.listdir(root)
        if osp.isfile(osp.join(root, name)) and name.lower().endswith(".ply")
    )


def _natural_block_key(cloud_id: str):
    match = _BLOCK_ID_RE.match(cloud_id)
    if match is None:
        return cloud_id, -1, cloud_id
    return match.group("city"), int(match.group("block")), cloud_id


def _city_fixed_splits(cloud_ids: List[str]) -> Dict[str, List[str]]:
    grouped = {city: [] for city in SENSATURBAN_CITY_SPLIT_COUNTS}
    unknown = []
    for cloud_id in cloud_ids:
        match = _BLOCK_ID_RE.match(cloud_id)
        if match is None:
            unknown.append(cloud_id)
            continue
        grouped[match.group("city")].append(cloud_id)

    splits = {"train": [], "val": [], "test": []}
    errors = []
    for city, counts in SENSATURBAN_CITY_SPLIT_COUNTS.items():
        ids = sorted(grouped[city], key=_natural_block_key)
        expected = sum(counts.values())
        if len(ids) != expected:
            errors.append(f"{city}: expected {expected}, found {len(ids)}")
            continue
        train_end = counts["train"]
        val_end = train_end + counts["val"]
        splits["train"].extend(ids[:train_end])
        splits["val"].extend(ids[train_end:val_end])
        splits["test"].extend(ids[val_end:])

    if unknown:
        errors.append(f"unknown file names: {unknown}")
    if errors:
        raise RuntimeError(
            "Cannot apply SensatUrban city_fixed split protocol. "
            "Expected birmingham_block_* and cambridge_block_* files with "
            "Birmingham 10/2/2 and Cambridge 18/4/4 counts. "
            + "; ".join(errors)
        )
    return splits


def _normalise_cloud_list(value) -> Optional[List[str]]:
    if value is None:
        return None
    return sorted(str(x) for x in value)


def _select_val_from_train(
        train_ids: List[str],
        val_ratio: float,
        val_count: Optional[int]) -> List[str]:
    if len(train_ids) <= 1:
        return []

    if val_count is None:
        val_count = int(round(len(train_ids) * float(val_ratio)))
        val_count = max(1, val_count)
    val_count = max(0, min(int(val_count), len(train_ids) - 1))
    if val_count == 0:
        return []

    step = len(train_ids) / float(val_count + 1)
    selected_indices = []
    used = set()
    for i in range(val_count):
        center = int(round((i + 1) * step))
        candidates = [center]
        for delta in range(1, len(train_ids)):
            candidates.extend([center - delta, center + delta])
        for idx in candidates:
            if 0 <= idx < len(train_ids) and idx not in used:
                selected_indices.append(idx)
                used.add(idx)
                break
    return [train_ids[i] for i in sorted(selected_indices)]


def resolve_sensaturban_splits(
        root: str,
        train_clouds=None,
        val_clouds=None,
        test_clouds=None,
        exclude_clouds=None,
        split_protocol: str = "auto",
        val_from_train: bool = True,
        val_ratio: float = 0.2,
        val_count: Optional[int] = None) -> Dict[str, List[str]]:
    """Resolve train/val/test ids without moving raw files on disk.

    The current project protocol is `city_fixed`: flat files are split as
    Birmingham 10/2/2 and Cambridge 18/4/4 by natural block order after
    pruning corrupted Cambridge blocks. The legacy directory protocol remains
    available for train/val/test folders.
    """
    if split_protocol not in ("auto", "city_fixed", "directory"):
        raise ValueError(
            f"Unknown SensatUrban split_protocol={split_protocol!r}. "
            "Expected one of: auto, city_fixed, directory."
        )

    overrides = {
        "train": _normalise_cloud_list(train_clouds),
        "val": _normalise_cloud_list(val_clouds),
        "test": _normalise_cloud_list(test_clouds),
    }
    has_split_override = any(ids is not None for ids in overrides.values())
    flat_ids = _discover_flat_cloud_ids(root)
    use_city_fixed = not has_split_override and (
        split_protocol == "city_fixed" or (split_protocol == "auto" and flat_ids))
    splits = _city_fixed_splits(flat_ids) if use_city_fixed else discover_sensaturban_splits(root)
    for stage, ids in overrides.items():
        if ids is not None:
            splits[stage] = ids

    excluded = set(_normalise_cloud_list(exclude_clouds) or [])
    if excluded:
        splits = {
            stage: [cloud_id for cloud_id in ids if cloud_id not in excluded]
            for stage, ids in splits.items()
        }

    explicit_val = overrides["val"] is not None
    if (not use_city_fixed and val_from_train
            and not explicit_val and not splits["val"] and splits["train"]):
        val_ids = _select_val_from_train(splits["train"], val_ratio, val_count)
        val_set = set(val_ids)
        splits["train"] = [cloud_id for cloud_id in splits["train"] if cloud_id not in val_set]
        splits["val"] = val_ids

    return splits


def _read_ply_vertex_layout(filepath: str) -> Tuple[int, List[Tuple[str, str]], int]:
    with open(filepath, "rb") as handle:
        header_size = 0
        vertex_count: Optional[int] = None
        vertex_properties: List[Tuple[str, str]] = []
        in_vertex = False
        binary_little_endian = False

        while True:
            line = handle.readline()
            if not line:
                raise RuntimeError(f"SensatUrban PLY {filepath} ended before end_header.")
            header_size += len(line)
            text = line.decode("ascii", errors="replace").strip()

            if text == "format binary_little_endian 1.0":
                binary_little_endian = True
            elif text.startswith("element "):
                parts = text.split()
                in_vertex = len(parts) == 3 and parts[1] == "vertex"
                if in_vertex:
                    vertex_count = int(parts[2])
            elif in_vertex and text.startswith("property "):
                parts = text.split()
                if len(parts) != 3:
                    raise RuntimeError(
                        f"SensatUrban PLY {filepath} has unsupported vertex property: {text}"
                    )
                _, dtype, name = parts
                vertex_properties.append((name, dtype))
            elif text == "end_header":
                break

    if not binary_little_endian:
        raise RuntimeError(
            f"SensatUrban PLY {filepath} must be binary_little_endian 1.0."
        )
    if vertex_count is None:
        raise RuntimeError(f"SensatUrban PLY {filepath} misses element vertex.")
    return vertex_count, vertex_properties, header_size


def validate_sensaturban_ply(filepath: str) -> None:
    """Fail before loading if the binary payload is inconsistent."""
    vertex_count, properties, header_size = _read_ply_vertex_layout(filepath)
    try:
        record_size = sum(_PLY_TYPE_BYTES[dtype] for _, dtype in properties)
    except KeyError as exc:
        raise RuntimeError(
            f"SensatUrban PLY {filepath} has unsupported scalar dtype {exc.args[0]}."
        ) from exc

    expected_size = header_size + vertex_count * record_size
    actual_size = osp.getsize(filepath)
    if expected_size != actual_size:
        raise RuntimeError(
            f"SensatUrban PLY size mismatch for {filepath}: "
            f"header declares {vertex_count} vertices with {record_size} bytes each "
            f"(expected {expected_size} bytes), but file has {actual_size} bytes. "
            "The file is likely truncated or has an invalid header."
        )


def _vertex_dtype(properties: List[Tuple[str, str]]) -> np.dtype:
    try:
        return np.dtype([(name, _PLY_NUMPY_DTYPES[dtype]) for name, dtype in properties])
    except KeyError as exc:
        raise RuntimeError(
            f"SensatUrban PLY has unsupported scalar dtype {exc.args[0]}."
        ) from exc


def _find_sensaturban_raw_path(root: str, cloud_id: str, stage: Optional[str] = None) -> str:
    candidates = [osp.join(root, cloud_id + ".ply")]
    if stage and stage != "trainval":
        candidates.append(osp.join(root, stage, cloud_id + ".ply"))
    candidates.extend(osp.join(root, s, cloud_id + ".ply") for s in ("train", "val", "test"))
    for path in candidates:
        if osp.exists(path):
            return path
    raise FileNotFoundError(
        f"Cannot find SensatUrban raw cloud {cloud_id!r} under {root}. "
        "Expected either flat PLY files or train/val/test subdirectories."
    )


def read_sensaturban_ply_stats(filepath: str, chunk_size: int = 5_000_000) -> Dict:
    """Stream a binary SensatUrban PLY and return bbox/point-count metadata."""
    validate_sensaturban_ply(filepath)
    vertex_count, properties, header_size = _read_ply_vertex_layout(filepath)
    dtype = _vertex_dtype(properties)
    names = set(dtype.names or [])
    missing = [axis for axis in ("x", "y", "z") if axis not in names]
    if missing:
        raise RuntimeError(
            f"SensatUrban cloud {filepath} misses coordinate fields {missing}. "
            "Expected x, y, z."
        )

    if vertex_count == 0:
        bbox_min = np.zeros(3, dtype=np.float64)
        bbox_max = np.zeros(3, dtype=np.float64)
    else:
        chunk_size = max(1, int(chunk_size))
        vertex = np.memmap(
            filepath,
            dtype=dtype,
            mode="r",
            offset=header_size,
            shape=(vertex_count,),
        )
        bbox_min = np.full(3, np.inf, dtype=np.float64)
        bbox_max = np.full(3, -np.inf, dtype=np.float64)
        for start in range(0, vertex_count, chunk_size):
            chunk = vertex[start:start + chunk_size]
            for i, axis in enumerate(("x", "y", "z")):
                values = np.asarray(chunk[axis], dtype=np.float64)
                bbox_min[i] = min(bbox_min[i], float(values.min()))
                bbox_max[i] = max(bbox_max[i], float(values.max()))
        del vertex

    extent = np.maximum(bbox_max - bbox_min, 0.0)
    stat = os.stat(filepath)
    return {
        "num_points": int(vertex_count),
        "bbox_min": bbox_min.tolist(),
        "bbox_max": bbox_max.tolist(),
        "extent": extent.tolist(),
        "width": float(extent[0]),
        "height": float(extent[1]),
        "source_size": int(stat.st_size),
        "source_mtime_ns": int(stat.st_mtime_ns),
    }


def compute_sensaturban_occupied_tiles(
        filepath: str,
        tiling: Tuple[int, int],
        stats: Dict,
        chunk_size: int = 5_000_000,
        min_tile_points: int = 1) -> Tuple[List[List[int]], List[List[int]]]:
    """Return 1-based XY tile coordinates containing at least one point.

    The computation mirrors `SampleXYTiling` so the manifest does not
    schedule tiles that would become empty immediately before GridSampling3D.
    """
    tx, ty = [int(v) for v in tiling]
    min_tile_points = max(1, int(min_tile_points))
    if tx <= 1 and ty <= 1:
        if int(stats.get("num_points", 0)) >= min_tile_points:
            return [[1, 1]], [[1, 1, int(stats.get("num_points", 0))]]
        return [], []

    vertex_count, properties, header_size = _read_ply_vertex_layout(filepath)
    if vertex_count == 0:
        return [], []
    dtype = _vertex_dtype(properties)
    names = set(dtype.names or [])
    missing = [axis for axis in ("x", "y") if axis not in names]
    if missing:
        raise RuntimeError(
            f"SensatUrban cloud {filepath} misses coordinate fields {missing}. "
            "Expected x, y."
        )

    bbox_min = np.asarray(stats["bbox_min"][:2], dtype=np.float64)
    extent = np.asarray(stats["extent"][:2], dtype=np.float64)
    extent = np.maximum(extent, np.finfo(np.float64).eps)
    point_counts = np.zeros((tx, ty), dtype=np.int64)
    chunk_size = max(1, int(chunk_size))
    vertex = np.memmap(
        filepath,
        dtype=dtype,
        mode="r",
        offset=header_size,
        shape=(vertex_count,),
    )
    tiling_arr = np.asarray([tx, ty], dtype=np.float64)
    for start in range(0, vertex_count, chunk_size):
        chunk = vertex[start:start + chunk_size]
        xy = np.stack([
            np.asarray(chunk["x"], dtype=np.float64),
            np.asarray(chunk["y"], dtype=np.float64),
        ], axis=1)
        grid = np.clip((xy - bbox_min.reshape(1, 2)) / extent.reshape(1, 2), 0.0, 1.0)
        grid = (grid * tiling_arr.reshape(1, 2)).astype(np.int64)

        # SampleXYTiling currently drops exact max-boundary points because
        # they map to index == tiling. Ignore them here too, otherwise a
        # manifest tile could still be empty after the actual transform.
        valid = (
            (grid[:, 0] >= 0) & (grid[:, 0] < tx)
            & (grid[:, 1] >= 0) & (grid[:, 1] < ty)
        )
        if valid.any():
            flat = grid[valid, 0] * ty + grid[valid, 1]
            counts = np.bincount(flat, minlength=tx * ty).reshape(tx, ty)
            point_counts += counts
    del vertex

    occupied_tiles = [
        [int(x + 1), int(y + 1)]
        for x in range(tx)
        for y in range(ty)
        if point_counts[x, y] >= min_tile_points
    ]
    tile_point_counts = [
        [int(x + 1), int(y + 1), int(point_counts[x, y])]
        for x in range(tx)
        for y in range(ty)
        if point_counts[x, y] > 0
    ]
    return occupied_tiles, tile_point_counts


def compute_adaptive_xy_tiling(
        width: float,
        height: float,
        tile_size_xy: Tuple[float, float],
        max_tiles_per_axis: int = 12,
        min_tiles_per_axis: int = 1) -> Tuple[int, int]:
    """Compute a per-cloud rectangular XY tiling from physical extents."""
    tile_x, tile_y = tile_size_xy
    if tile_x <= 0 or tile_y <= 0:
        raise ValueError(f"tile_size_xy must be positive, got {tile_size_xy}.")
    max_tiles_per_axis = max(1, int(max_tiles_per_axis))
    min_tiles_per_axis = max(1, int(min_tiles_per_axis))
    if min_tiles_per_axis > max_tiles_per_axis:
        raise ValueError(
            f"min_tiles_per_axis={min_tiles_per_axis} exceeds "
            f"max_tiles_per_axis={max_tiles_per_axis}."
        )

    nx = int(math.ceil(max(float(width), 0.0) / float(tile_x))) if width > 0 else 1
    ny = int(math.ceil(max(float(height), 0.0) / float(tile_y))) if height > 0 else 1
    nx = min(max(nx, min_tiles_per_axis), max_tiles_per_axis)
    ny = min(max(ny, min_tiles_per_axis), max_tiles_per_axis)
    return nx, ny


def _unique_split_ids(splits: Dict[str, List[str]]) -> List[Tuple[str, str]]:
    seen = set()
    out = []
    for stage in ("train", "val", "test"):
        for cloud_id in splits.get(stage, []):
            if cloud_id in seen:
                continue
            seen.add(cloud_id)
            out.append((stage, cloud_id))
    return out


def _adaptive_config_get(config, key: str, default=None):
    if config is None:
        return default
    if hasattr(config, "get"):
        return config.get(key, default)
    return getattr(config, key, default)


def _adaptive_manifest_path(root: str, config) -> str:
    path = _adaptive_config_get(config, "manifest_path")
    if path:
        return path if osp.isabs(path) else osp.join(root, path)
    name = _adaptive_config_get(
        config,
        "manifest_name",
        "adaptive_xy_tiling_manifest.json",
    )
    return osp.join(root, "metadata", name)


def _adaptive_manifest_is_valid(
        root: str,
        manifest: Dict,
        splits: Dict[str, List[str]],
        config) -> bool:
    if manifest.get("version") != 3:
        return False
    clouds = manifest.get("clouds", {})
    for stage, cloud_id in _unique_split_ids(splits):
        entry = clouds.get(cloud_id)
        if entry is None:
            return False
        if "occupied_tiles" not in entry:
            return False
        if "tile_point_counts" not in entry:
            return False
        path = _find_sensaturban_raw_path(root, cloud_id, stage=stage)
        stat = os.stat(path)
        if int(entry.get("source_size", -1)) != int(stat.st_size):
            return False
        if int(entry.get("source_mtime_ns", -1)) != int(stat.st_mtime_ns):
            return False
    reference_cloud = _adaptive_config_get(config, "reference_cloud")
    reference_xy_tiling = int(_adaptive_config_get(config, "reference_xy_tiling", 0) or 0)
    manifest_config = manifest.get("config", {})
    min_tile_points = int(_adaptive_config_get(config, "min_tile_points", 1))
    return (
        manifest_config.get("reference_cloud") == reference_cloud
        and int(manifest_config.get("reference_xy_tiling", 0) or 0) == reference_xy_tiling
        and int(manifest_config.get("min_tile_points", 1) or 1) == min_tile_points
    )


def build_sensaturban_adaptive_tiling_manifest(
        root: str,
        splits: Dict[str, List[str]],
        config) -> Dict:
    """Create or load per-cloud adaptive XY tiling metadata for SensatUrban."""
    manifest_path = _adaptive_manifest_path(root, config)
    force_rebuild = bool(_adaptive_config_get(config, "force_rebuild", False))
    if osp.exists(manifest_path) and not force_rebuild:
        with open(manifest_path, "r", encoding="utf-8") as handle:
            manifest = json.load(handle)
        if _adaptive_manifest_is_valid(root, manifest, splits, config):
            return manifest

    reference_cloud = _adaptive_config_get(config, "reference_cloud", "birmingham_block_9")
    reference_xy_tiling = int(_adaptive_config_get(config, "reference_xy_tiling", 3))
    if reference_xy_tiling < 1:
        raise ValueError("adaptive_xy_tiling.reference_xy_tiling must be >= 1.")
    chunk_size = int(_adaptive_config_get(config, "chunk_size", 5_000_000))
    max_tiles_per_axis = int(_adaptive_config_get(config, "max_tiles_per_axis", 12))
    min_tiles_per_axis = int(_adaptive_config_get(config, "min_tiles_per_axis", 1))
    min_tile_size = float(_adaptive_config_get(config, "min_tile_size", 1.0))
    min_tile_points = int(_adaptive_config_get(config, "min_tile_points", 1))
    target_tile_size = _adaptive_config_get(config, "target_tile_size")
    target_tile_size_xy = _adaptive_config_get(config, "target_tile_size_xy")

    stats = {}
    stage_by_id = dict((cloud_id, stage) for stage, cloud_id in _unique_split_ids(splits))
    if reference_cloud not in stage_by_id:
        stage_by_id[reference_cloud] = None

    reference_path = _find_sensaturban_raw_path(
        root,
        reference_cloud,
        stage=stage_by_id.get(reference_cloud),
    )
    reference_stats = read_sensaturban_ply_stats(reference_path, chunk_size=chunk_size)
    stats[reference_cloud] = reference_stats

    if target_tile_size_xy is not None:
        tile_size_xy = tuple(float(x) for x in target_tile_size_xy)
    elif target_tile_size is not None:
        size = float(target_tile_size)
        tile_size_xy = (size, size)
    else:
        tile_size_xy = (
            max(float(reference_stats["width"]) / reference_xy_tiling, min_tile_size),
            max(float(reference_stats["height"]) / reference_xy_tiling, min_tile_size),
        )

    clouds = {}
    for stage, cloud_id in _unique_split_ids(splits):
        if cloud_id in stats:
            cloud_stats = stats[cloud_id]
        else:
            path = _find_sensaturban_raw_path(root, cloud_id, stage=stage)
            cloud_stats = read_sensaturban_ply_stats(path, chunk_size=chunk_size)
            stats[cloud_id] = cloud_stats
        nx, ny = compute_adaptive_xy_tiling(
            cloud_stats["width"],
            cloud_stats["height"],
            tile_size_xy,
            max_tiles_per_axis=max_tiles_per_axis,
            min_tiles_per_axis=min_tiles_per_axis,
        )
        path = _find_sensaturban_raw_path(root, cloud_id, stage=stage)
        occupied_tiles, tile_point_counts = compute_sensaturban_occupied_tiles(
            path,
            (nx, ny),
            cloud_stats,
            chunk_size=chunk_size,
            min_tile_points=min_tile_points,
        )
        clouds[cloud_id] = {
            **cloud_stats,
            "tiling": [int(nx), int(ny)],
            "grid_tile_count": int(nx * ny),
            "occupied_tiles": occupied_tiles,
            "tile_point_counts": tile_point_counts,
            "tile_count": len(occupied_tiles),
        }

    manifest = {
        "version": 3,
        "config": {
            "reference_cloud": reference_cloud,
            "reference_xy_tiling": reference_xy_tiling,
            "target_tile_size_xy": [float(tile_size_xy[0]), float(tile_size_xy[1])],
            "max_tiles_per_axis": max_tiles_per_axis,
            "min_tiles_per_axis": min_tiles_per_axis,
            "min_tile_size": min_tile_size,
            "min_tile_points": min_tile_points,
        },
        "reference": {
            "cloud_id": reference_cloud,
            "stats": reference_stats,
        },
        "clouds": clouds,
    }
    os.makedirs(osp.dirname(manifest_path), exist_ok=True)
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
    return manifest


def expand_sensaturban_adaptive_cloud_ids(
        splits: Dict[str, List[str]],
        manifest: Dict) -> Dict[str, List[str]]:
    """Expand base cloud ids according to a per-cloud adaptive tiling manifest."""
    out = {}
    clouds = manifest.get("clouds", {})
    for stage, ids in splits.items():
        stage_ids = []
        for cloud_id in ids:
            entry = clouds.get(cloud_id)
            if entry is None:
                raise KeyError(f"Missing adaptive tiling manifest entry for {cloud_id}.")
            tx, ty = [int(v) for v in entry.get("tiling", [1, 1])]
            occupied_tiles = entry.get("occupied_tiles")
            if occupied_tiles is None:
                occupied_tiles = [
                    [x + 1, y + 1]
                    for x in range(tx)
                    for y in range(ty)
                ]
            if tx <= 1 and ty <= 1 and occupied_tiles:
                stage_ids.append(cloud_id)
                continue
            stage_ids.extend(
                f"{cloud_id}__TILE_{int(x)}-{int(y)}_OF_{tx}-{ty}"
                for x, y in occupied_tiles
            )
        out[stage] = stage_ids
    return out


def _vertex_array(vertex, key: str) -> np.ndarray:
    return np.asarray(vertex[key]).copy()


def read_sensaturban_cloud(
        raw_cloud_path: str,
        xyz: bool = True,
        rgb: bool = True,
        semantic: bool = True,
        remap: bool = True,
        allow_unlabelled: bool = False) -> Data:
    """Read one SensatUrban PLY cloud.

    Train/val files are expected to carry `class` labels in the official
    13-class SensatUrban id space. Test files may omit labels.
    """
    validate_sensaturban_ply(raw_cloud_path)

    data = Data()
    with open(raw_cloud_path, "rb") as handle:
        ply = PlyData.read(handle)
        vertex = ply["vertex"]
        attributes = set(vertex.data.dtype.names or [])

        if xyz:
            missing = [axis for axis in ("x", "y", "z") if axis not in attributes]
            if missing:
                raise RuntimeError(
                    f"SensatUrban cloud {raw_cloud_path} misses coordinate fields {missing}. "
                    "Expected x, y, z."
                )
            pos = torch.stack([
                torch.as_tensor(_vertex_array(vertex, axis), dtype=torch.float32)
                for axis in ("x", "y", "z")
            ], dim=-1)
            pos_offset = pos[0].clone()
            data.pos = pos - pos_offset
            data.pos_offset = pos_offset

        if rgb:
            rgb_keys = ("red", "green", "blue")
            if not all(key in attributes for key in rgb_keys):
                missing = [key for key in rgb_keys if key not in attributes]
                raise RuntimeError(
                    f"SensatUrban cloud {raw_cloud_path} misses RGB fields {missing}. "
                    "Expected red, green, blue."
                )
            data.rgb = to_float_rgb(torch.stack([
                torch.as_tensor(_vertex_array(vertex, key), dtype=torch.float32)
                for key in rgb_keys
            ], dim=-1))

        if semantic:
            if "class" not in attributes:
                if allow_unlabelled:
                    return data
                raise RuntimeError(
                    f"SensatUrban cloud {raw_cloud_path} misses semantic labels. "
                    "Expected a uint8 `class` property for train/val clouds."
                )
            raw_label = np.rint(_vertex_array(vertex, "class")).astype(np.int64, copy=False)
            invalid = (raw_label < 0) | (raw_label >= SENSATURBAN_NUM_CLASSES)
            mapped = raw_label.copy()
            if remap:
                mapped[~invalid] = ID2TRAINID[mapped[~invalid]]
            mapped[invalid] = SENSATURBAN_NUM_CLASSES
            data.y = torch.as_tensor(mapped, dtype=torch.long)

    return data


class SensatUrban(BaseDataset):
    """SensatUrban semantic segmentation dataset adapter."""

    def __init__(
        self,
        root: str,
        *args,
        train_clouds=None,
        val_clouds=None,
        test_clouds=None,
        exclude_clouds=None,
        split_protocol: str = "auto",
        val_from_train: bool = True,
        val_ratio: float = 0.2,
        val_count: Optional[int] = None,
        adaptive_xy_tiling=None,
        **kwargs):
        self.train_clouds = train_clouds
        self.val_clouds = val_clouds
        self.test_clouds = test_clouds
        self.exclude_clouds = exclude_clouds
        self.split_protocol = split_protocol
        self.val_from_train = val_from_train
        self.val_ratio = val_ratio
        self.val_count = val_count
        self.adaptive_xy_tiling = adaptive_xy_tiling
        self._adaptive_xy_tiling_manifest_cache = None
        self._sensaturban_root = osp.join(root, self.data_subdir_name)
        super().__init__(root, *args, **kwargs)

    @property
    def class_names(self) -> List[str]:
        return CLASS_NAMES

    @property
    def num_classes(self) -> int:
        return SENSATURBAN_NUM_CLASSES

    @property
    def stuff_classes(self) -> List[int]:
        return STUFF_CLASSES

    @property
    def class_colors(self) -> List[List[int]]:
        return CLASS_COLORS

    @property
    def data_subdir_name(self) -> str:
        return "SensatUrban"

    @property
    def raw_dir(self) -> str:
        return self.root

    @property
    def all_base_cloud_ids(self) -> Dict[str, List[str]]:
        return resolve_sensaturban_splits(
            self._sensaturban_root,
            train_clouds=self.train_clouds,
            val_clouds=self.val_clouds,
            test_clouds=self.test_clouds,
            exclude_clouds=self.exclude_clouds,
            split_protocol=self.split_protocol,
            val_from_train=self.val_from_train,
            val_ratio=self.val_ratio,
            val_count=self.val_count,
        )

    @property
    def adaptive_xy_tiling_enabled(self) -> bool:
        return bool(_adaptive_config_get(self.adaptive_xy_tiling, "enable", False))

    @property
    def adaptive_xy_tiling_manifest(self) -> Optional[Dict]:
        if not self.adaptive_xy_tiling_enabled:
            return None
        if self._adaptive_xy_tiling_manifest_cache is None:
            self._adaptive_xy_tiling_manifest_cache = build_sensaturban_adaptive_tiling_manifest(
                self._sensaturban_root,
                self.all_base_cloud_ids,
                self.adaptive_xy_tiling,
            )
        return self._adaptive_xy_tiling_manifest_cache

    @property
    def all_cloud_ids(self) -> Dict[str, List[str]]:
        if self.adaptive_xy_tiling_enabled:
            return expand_sensaturban_adaptive_cloud_ids(
                self.all_base_cloud_ids,
                self.adaptive_xy_tiling_manifest,
            )
        return super().all_cloud_ids

    @property
    def raw_file_structure(self) -> str:
        return f"""
    {self.root}/
        ├── birmingham_block_{{0..13}}.ply
        └── cambridge_block_{{...}}.ply

    Legacy train/val/test subdirectories are also supported when
    split_protocol='directory'.
            """

    def download_dataset(self) -> None:
        raise RuntimeError(
            "SensatUrban automatic download is not implemented. Place PLY files under "
            "`data/SensatUrban/train`, optionally `data/SensatUrban/val`, and "
            "`data/SensatUrban/test`. Train/val files must contain x/y/z, "
            "red/green/blue, and class fields."
        )

    def _raw_stage_for_base_id(self, stage: str, base_id: str) -> str:
        flat_path = osp.join(self.raw_dir, base_id + ".ply")
        if osp.exists(flat_path):
            return ""

        preferred = osp.join(self.raw_dir, stage, base_id + ".ply")
        if stage != "trainval" and osp.exists(preferred):
            return stage

        train_path = osp.join(self.raw_dir, "train", base_id + ".ply")
        if osp.exists(train_path):
            return "train"

        val_path = osp.join(self.raw_dir, "val", base_id + ".ply")
        if osp.exists(val_path):
            return "val"

        return "train" if stage in ("train", "val", "trainval") else stage

    def id_to_relative_raw_path(self, id: str) -> str:
        base_id = self.id_to_base_id(id)
        for stage in ("train", "val", "test"):
            if id in self.all_cloud_ids[stage]:
                raw_stage = self._raw_stage_for_base_id(stage, base_id)
                return osp.join(raw_stage, base_id + ".ply") if raw_stage else base_id + ".ply"
        raise ValueError(f"Unknown SensatUrban cloud id '{id}'")

    def processed_to_raw_path(self, processed_path: str) -> str:
        stage, _, cloud_id = osp.splitext(processed_path)[0].split(os.sep)[-3:]
        base_cloud_id = self.id_to_base_id(cloud_id)
        raw_stage = self._raw_stage_for_base_id(stage, base_cloud_id)
        return osp.join(self.raw_dir, raw_stage, base_cloud_id + ".ply") \
            if raw_stage else osp.join(self.raw_dir, base_cloud_id + ".ply")

    def read_single_raw_cloud(self, raw_cloud_path: str) -> Data:
        return read_sensaturban_cloud(
            raw_cloud_path,
            semantic=True,
            allow_unlabelled=self.stage == "test")


class MiniSensatUrban(SensatUrban):
    _NUM_MINI = 1

    @property
    def all_cloud_ids(self) -> Dict[str, List[str]]:
        return {k: v[:self._NUM_MINI] for k, v in super().all_cloud_ids.items()}

    @property
    def data_subdir_name(self) -> str:
        return "SensatUrban"

    def process(self) -> None:
        super().process()

    def download(self) -> None:
        super().download()
