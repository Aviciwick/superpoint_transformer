import os.path as osp
from typing import List

import numpy as np
import torch
from plyfile import PlyData

from src.data import Data
from src.datasets import BaseDataset
from src.datasets.toronto3d_config import *
from src.utils.color import to_float_rgb


__all__ = ["Toronto3D", "MiniToronto3D"]


def _first_existing_property(vertex, names):
    available = set(vertex.data.dtype.names or [])
    for name in names:
        if name in available:
            return name
    return None


def _vertex_array(vertex, key: str) -> np.ndarray:
    return np.asarray(vertex[key]).copy()


def read_toronto3d_cloud(
        raw_cloud_path: str,
        xyz: bool = True,
        rgb: bool = True,
        intensity: bool = True,
        semantic: bool = True,
        remap: bool = True) -> Data:
    """Read one Toronto-3D PLY cloud.

    Expected project files contain `x/y/z`, `red/green/blue`,
    `scalar_Intensity`, and `scalar_Label`. Raw labels follow the
    9-class Toronto-3D convention: 0 is unclassified/void, labels 1-8
    are mapped to the 8 train classes used by the paper.
    """
    data = Data()
    with open(raw_cloud_path, "rb") as handle:
        ply = PlyData.read(handle)
        vertex = ply["vertex"]
        attributes = set(vertex.data.dtype.names or [])

        if xyz:
            missing = [axis for axis in ("x", "y", "z") if axis not in attributes]
            if missing:
                raise RuntimeError(
                    f"Toronto-3D cloud {raw_cloud_path} misses coordinate fields {missing}. "
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
            if all(key in attributes for key in rgb_keys):
                data.rgb = to_float_rgb(torch.stack([
                    torch.as_tensor(_vertex_array(vertex, key), dtype=torch.float32)
                    for key in rgb_keys
                ], dim=-1))

        if intensity:
            intensity_key = _first_existing_property(
                vertex,
                ("scalar_Intensity", "intensity", "Intensity"),
            )
            if intensity_key is not None:
                values = torch.as_tensor(_vertex_array(vertex, intensity_key), dtype=torch.float32)
                finite = torch.isfinite(values)
                scale = values[finite].max().clamp(min=1.0) if finite.any() else values.new_tensor(1.0)
                values = torch.where(finite, values, values.new_zeros(()))
                data.intensity = (values / scale).clamp(min=0.0, max=1.0).view(-1, 1)

        if semantic:
            label_key = _first_existing_property(
                vertex,
                ("scalar_Label", "label", "Label", "class", "semantic"),
            )
            if label_key is None:
                raise RuntimeError(
                    f"Toronto-3D cloud {raw_cloud_path} misses semantic labels. "
                    "Expected scalar_Label with raw ids 0-8."
                )
            raw_label = np.rint(_vertex_array(vertex, label_key)).astype(np.int64, copy=False)
            invalid = (raw_label < 0) | (raw_label >= ID2TRAINID.shape[0])
            raw_label = raw_label.copy()
            raw_label[invalid] = 0
            mapped = ID2TRAINID[raw_label] if remap else raw_label
            data.y = torch.as_tensor(mapped, dtype=torch.long)

    return data


class Toronto3D(BaseDataset):
    """Toronto-3D dataset adapter for the project-local PLY package."""

    def __init__(self, *args, train_clouds=None, test_clouds=None, **kwargs):
        self.train_clouds = train_clouds
        self.test_clouds = test_clouds
        super().__init__(*args, **kwargs)

    @property
    def class_names(self) -> List[str]:
        return CLASS_NAMES

    @property
    def num_classes(self) -> int:
        return TORONTO3D_NUM_CLASSES

    @property
    def stuff_classes(self) -> List[int]:
        return STUFF_CLASSES

    @property
    def class_colors(self) -> List[List[int]]:
        return CLASS_COLORS

    @property
    def all_base_cloud_ids(self):
        splits = TORONTO3D_SPLITS.copy()
        if self.train_clouds is not None:
            splits["train"] = self.train_clouds
        if self.test_clouds is not None:
            splits["test"] = self.test_clouds
        return splits

    @property
    def data_subdir_name(self) -> str:
        return "Toronto_3D"

    @property
    def raw_dir(self) -> str:
        # The provided package stores L001-L004.ply directly under
        # data/Toronto_3D rather than data/Toronto_3D/raw.
        return self.root

    @property
    def raw_file_structure(self) -> str:
        return f"""
    {self.root}/
        ├── L001.ply
        ├── L002.ply
        ├── L003.ply
        ├── L004.ply
        ├── Colors.xml
        └── toronto_3d_classes_9.txt
            """

    def download_dataset(self) -> None:
        raise RuntimeError(
            "Toronto-3D data is not bundled and automatic download is not implemented. "
            "Place L001.ply, L002.ply, L003.ply, and L004.ply directly under "
            "`data/Toronto_3D`. Required fields are x/y/z, red/green/blue, "
            "scalar_Intensity, and scalar_Label with raw ids 0-8."
        )

    def id_to_relative_raw_path(self, id: str) -> str:
        return f"{self.id_to_base_id(id)}.ply"

    def processed_to_raw_path(self, processed_path: str) -> str:
        stage, hash_dir, cloud_id = osp.splitext(processed_path)[0].split(osp.sep)[-3:]
        del stage, hash_dir
        return osp.join(self.raw_dir, f"{self.id_to_base_id(cloud_id)}.ply")

    def read_single_raw_cloud(self, raw_cloud_path: str) -> Data:
        return read_toronto3d_cloud(raw_cloud_path)


class MiniToronto3D(Toronto3D):
    _NUM_MINI = 1

    @property
    def all_cloud_ids(self):
        return {k: v[:self._NUM_MINI] for k, v in super().all_cloud_ids.items()}

    @property
    def data_subdir_name(self) -> str:
        return "Toronto_3D"

    def process(self) -> None:
        super().process()

    def download(self) -> None:
        super().download()
