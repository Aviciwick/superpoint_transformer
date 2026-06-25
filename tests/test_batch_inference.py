import json
import os
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch.utils.data import ConcatDataset, Dataset


project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, "src"))

import batch_inference


class _DummyDataset(Dataset):
    def __init__(self, root=None, cloud_ids=None, *, on_device_transform=None, stage=None, transform=None, pre_transform=None, **kwargs):
        if cloud_ids is None and isinstance(root, (list, tuple)):
            cloud_ids = list(root)
            root = None
        if cloud_ids is None:
            cloud_ids = []
        self.root = root
        self.cloud_ids = cloud_ids
        self.class_names = ["ground", "car"]
        self.class_colors = np.asarray([[1, 2, 3], [4, 5, 6]])
        self.num_classes = 2
        self.stuff_classes = [0, 7]
        self.on_device_transform = on_device_transform
        self.stage = stage
        self.transform = transform
        self.pre_transform = pre_transform
        self.kwargs = kwargs

    def __len__(self):
        return len(self.cloud_ids)

    def __getitem__(self, idx):
        return {"dataset_cloud_id": self.cloud_ids[idx], "sample_idx": idx}


class _DummyNAG(list):
    @property
    def num_levels(self):
        return len(self)


def test_resolve_dataset_metadata_adds_void_label_and_color():
    dataset = _DummyDataset(["scene_a"])
    metadata = batch_inference.resolve_dataset_metadata(dataset)

    assert metadata["dataset_name"] == "_DummyDataset"
    assert metadata["num_classes"] == 2
    assert metadata["class_names"] == ["ground", "car", "void"]
    assert metadata["class_colors"] == [[1, 2, 3], [4, 5, 6], [0, 0, 0]]
    assert metadata["stuff_classes"] == [0]


def test_resolve_cloud_ids_supports_concat_dataset():
    dataset = ConcatDataset([
        _DummyDataset(["train_0", "train_1"]),
        _DummyDataset(["val_0"]),
    ])

    cloud_ids = batch_inference._resolve_cloud_ids(dataset)

    assert cloud_ids == ["train_0", "train_1", "val_0"]


def test_select_dataset_respects_requested_stage():
    train_dataset = _DummyDataset(["train_0"])
    val_dataset = _DummyDataset(["val_0"])
    test_dataset = _DummyDataset(["test_0"])
    datamodule = SimpleNamespace(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
    )

    assert batch_inference._select_dataset(datamodule, "train") is train_dataset
    assert batch_inference._select_dataset(datamodule, "val") is val_dataset
    assert batch_inference._select_dataset(datamodule, "test") is test_dataset

    trainval_dataset = batch_inference._select_dataset(datamodule, "trainval")
    assert isinstance(trainval_dataset, ConcatDataset)
    assert trainval_dataset.datasets == [train_dataset, val_dataset]


def test_build_inference_dataset_uses_exact_requested_stage_instead_of_datamodule_remap():
    datamodule = SimpleNamespace(
        dataset_class=_DummyDataset,
        hparams=SimpleNamespace(data_dir="/tmp/unused", val_on_test=True, trainval=True),
        kwargs={"cloud_ids": ["scene_a"]},
        pre_transform="pre",
        val_transform="val_transform",
        test_transform="test_transform",
        on_device_val_transform="val_odt",
        on_device_test_transform="test_odt",
    )

    dataset = batch_inference._build_inference_dataset(datamodule, "val")

    assert dataset.stage == "val"
    assert dataset.transform == "val_transform"
    assert dataset.on_device_transform == "val_odt"


def test_build_inference_dataset_uses_test_transforms_for_test_stage():
    datamodule = SimpleNamespace(
        dataset_class=_DummyDataset,
        hparams=SimpleNamespace(data_dir="/tmp/unused"),
        kwargs={"cloud_ids": ["scene_a"]},
        pre_transform="pre",
        val_transform="val_transform",
        test_transform="test_transform",
        on_device_val_transform="val_odt",
        on_device_test_transform="test_odt",
    )

    dataset = batch_inference._build_inference_dataset(datamodule, "test")

    assert dataset.stage == "test"
    assert dataset.transform == "test_transform"
    assert dataset.on_device_transform == "test_odt"


def test_resolve_dataset_sample_returns_dataset_specific_transform_for_concat_dataset():
    train_transform = object()
    val_transform = object()
    dataset = ConcatDataset([
        _DummyDataset(["train_0", "train_1"], on_device_transform=train_transform),
        _DummyDataset(["val_0"], on_device_transform=val_transform),
    ])

    sample_0, transform_0 = batch_inference._resolve_dataset_sample(dataset, 0)
    sample_2, transform_2 = batch_inference._resolve_dataset_sample(dataset, 2)

    assert sample_0["dataset_cloud_id"] == "train_0"
    assert transform_0 is train_transform
    assert sample_2["dataset_cloud_id"] == "val_0"
    assert transform_2 is val_transform


def test_collect_visualization_keys_returns_bsr_pointwise_diagnostics():
    nag = [
        SimpleNamespace(
            bsr_candidate_mask=1,
            bsr_candidate_score=1,
            bsr_assignment_entropy=None,
            bsr_secondary_slot_mass=1,
            bsr_dual_slot_active=None,
            bsr_slot_diversity=1,
        )
    ]

    keys = batch_inference._collect_visualization_keys(nag)

    assert keys == [
        "bsr_candidate_mask",
        "bsr_candidate_score",
        "bsr_secondary_slot_mass",
        "bsr_slot_diversity",
    ]


def test_ensure_inference_point_keys_adds_rgb():
    cfg = SimpleNamespace(
        datamodule=SimpleNamespace(
            point_load_keys=["pos", "y", "super_index"],
        )
    )

    batch_inference._ensure_inference_point_keys(cfg)

    assert cfg.datamodule.point_load_keys == ["pos", "y", "super_index", "rgb"]


def test_extract_checkpoint_signature_reads_experiment_hint(tmp_path):
    run_dir = tmp_path / "run"
    ckpt_dir = run_dir / "checkpoints"
    hydra_dir = run_dir / ".hydra"
    ckpt_dir.mkdir(parents=True)
    hydra_dir.mkdir(parents=True)
    ckpt_path = ckpt_dir / "epoch_169.ckpt"
    ckpt_path.write_text("", encoding="utf-8")
    (hydra_dir / "overrides.yaml").write_text(
        "- experiment=semantic/kitti360\n- seed=1\n",
        encoding="utf-8",
    )

    signature = batch_inference._extract_checkpoint_signature(
        {
            "hyper_parameters": {
                "num_classes": 15,
                "_down_dim": [128, 128, 128, 128],
                "_up_dim": [128, 128, 128],
                "_point_out_dim": 128,
                "_point_hf_dim": 132,
                "bsr": {"enable": False},
            }
        },
        str(ckpt_path),
    )

    assert signature["experiment_hint"] == "semantic/kitti360"
    assert signature["num_classes"] == 15
    assert signature["down_dim"] == [128, 128, 128, 128]
    assert signature["up_dim"] == [128, 128, 128]
    assert signature["point_out_dim"] == 128
    assert signature["point_hf_dim"] == 132
    assert signature["bsr_enable"] is False


def test_load_inference_config_prefers_checkpoint_hydra_config(tmp_path, monkeypatch):
    run_dir = tmp_path / "run"
    ckpt_dir = run_dir / "checkpoints"
    hydra_dir = run_dir / ".hydra"
    ckpt_dir.mkdir(parents=True)
    hydra_dir.mkdir(parents=True)
    ckpt_path = ckpt_dir / "epoch_001.ckpt"
    ckpt_path.write_text("", encoding="utf-8")
    (hydra_dir / "config.yaml").write_text(
        """
datamodule:
  point_hf:
    - rgb
  dataloader:
    batch_size: 4
    num_workers: 8
model:
  _point_hf_dim: 12
  bsr:
    enable: false
ckpt_path: null
""",
        encoding="utf-8",
    )

    def fail_if_current_config_is_used(*args, **kwargs):
        raise AssertionError("current experiment config should not be loaded when checkpoint config exists")

    monkeypatch.setattr(batch_inference, "init_config", fail_if_current_config_is_used)

    cfg, source = batch_inference._load_inference_config(
        experiment="semantic/toronto3d",
        ckpt_path=str(ckpt_path),
    )

    assert source == str(hydra_dir / "config.yaml")
    assert list(cfg.datamodule.point_hf) == ["rgb"]
    assert cfg.model._point_hf_dim == 12
    assert cfg.datamodule.dataloader.batch_size == 1
    assert cfg.datamodule.dataloader.num_workers == 0
    assert cfg.ckpt_path == str(ckpt_path)


def test_validate_checkpoint_compatibility_raises_human_readable_error():
    checkpoint_signature = {
        "experiment_hint": "semantic/kitti360",
        "num_classes": 15,
        "down_dim": [128, 128, 128, 128],
        "up_dim": [128, 128, 128],
        "point_out_dim": 128,
        "point_hf_dim": 132,
        "bsr_enable": False,
    }
    model_signature = {
        "experiment_hint": None,
        "num_classes": 20,
        "down_dim": [64, 64, 64, 64],
        "up_dim": [64, 64, 64],
        "point_out_dim": 64,
        "point_hf_dim": 68,
        "bsr_enable": True,
    }

    with pytest.raises(ValueError, match="Checkpoint/config mismatch detected before loading weights"):
        batch_inference._validate_checkpoint_compatibility(
            requested_experiment="semantic/scannet_bsr",
            checkpoint_signature=checkpoint_signature,
            model_signature=model_signature,
        )


def test_prepare_save_data_collects_visualizable_point_fields():
    nag = _DummyNAG([
        SimpleNamespace(
            pos=torch.tensor([[0.0, 0.1, 0.2], [1.0, 1.1, 1.2]], dtype=torch.float32),
            pos_offset=torch.tensor([10.0, 20.0, 30.0], dtype=torch.float32),
            rgb=torch.tensor([[0.1, 0.2, 0.3], [0.8, 0.7, 0.6]], dtype=torch.float32),
            semantic_pred=torch.tensor([1, 0], dtype=torch.long),
            y=torch.tensor([1, 1], dtype=torch.long),
            obj=torch.tensor([7, 8], dtype=torch.long),
            obj_pred=torch.tensor([70, 80], dtype=torch.long),
            x=torch.tensor([[0.1, 0.2], [0.3, 0.4]], dtype=torch.float32),
            super_sampling=torch.tensor([2, -1], dtype=torch.long),
            bsr_candidate_mask=torch.tensor([1, 0], dtype=torch.long),
            bsr_candidate_score=torch.tensor([0.9, 0.1], dtype=torch.float32),
            bsr_assignment_entropy=torch.tensor([0.2, 0.4], dtype=torch.float32),
            bsr_secondary_slot_mass=torch.tensor([0.6, 0.3], dtype=torch.float32),
            bsr_dual_slot_active=torch.tensor([1, 0], dtype=torch.long),
            bsr_slot_diversity=torch.tensor([0.5, 0.7], dtype=torch.float32),
            super_index=torch.tensor([3, 4], dtype=torch.long),
        ),
        SimpleNamespace(
            super_index=torch.tensor([0, 0, 1, 1, 1], dtype=torch.long),
        ),
    ])
    metadata = {
        "dataset_name": "DummySet",
        "class_names": ["ground", "car", "void"],
        "class_colors": [[1, 2, 3], [4, 5, 6], [0, 0, 0]],
        "num_classes": 2,
        "stuff_classes": [0],
    }

    save_data = batch_inference._prepare_save_data(nag, metadata)

    assert save_data["num_points"] == 2
    assert save_data["pos"].tolist() == [[10.0, 20.100000381469727, 30.200000762939453], [11.0, 21.100000381469727, 31.200000762939453]]
    assert "semantic_pred" in save_data["export_fields"]
    assert "gt_label" in save_data["export_fields"]
    assert "obj" in save_data["export_fields"]
    assert "obj_pred" in save_data["export_fields"]
    assert "sp_level_1" in save_data["export_fields"]
    assert "super_sampling" in save_data["export_fields"]
    assert "feature_vis_rgb_red" in save_data["export_fields"]
    assert "bsr_candidate_score" in save_data["export_fields"]
    assert "bsr_assignment_entropy" in save_data["export_fields"]
    assert save_data["metadata"]["export_fields"] == list(save_data["export_fields"].keys())


def test_prepare_save_data_accepts_repeated_chunked_pos_offset():
    nag = _DummyNAG([
        SimpleNamespace(
            pos=torch.tensor([[0.0, 0.1, 0.2], [1.0, 1.1, 1.2]], dtype=torch.float32),
            pos_offset=torch.tensor(
                [
                    10.0, 20.0, 30.0,
                    10.0, 20.0, 30.0,
                    10.0, 20.0, 30.0,
                ],
                dtype=torch.float32,
            ),
            rgb=None,
            semantic_pred=torch.tensor([1, 0], dtype=torch.long),
            y=None,
            obj=None,
            obj_pred=None,
            super_index=torch.tensor([0, 1], dtype=torch.long),
        ),
        SimpleNamespace(super_index=torch.tensor([0, 1], dtype=torch.long)),
    ])
    metadata = {
        "dataset_name": "DummySet",
        "class_names": ["ground", "car", "void"],
        "class_colors": [[1, 2, 3], [4, 5, 6], [0, 0, 0]],
        "num_classes": 2,
        "stuff_classes": [0],
    }

    save_data = batch_inference._prepare_save_data(nag, metadata)

    assert save_data["pos"].tolist() == [
        [10.0, 20.100000381469727, 30.200000762939453],
        [11.0, 21.100000381469727, 31.200000762939453],
    ]


def test_save_predictions_ply_writes_all_exported_properties_and_metadata(tmp_path):
    save_data = {
        "pos": np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], dtype=np.float32),
        "rgb": np.array([[255, 0, 0], [0, 255, 0]], dtype=np.uint8),
        "num_points": 2,
        "export_fields": {
            "semantic_pred": np.array([1, 0], dtype=np.int32),
            "obj_pred": np.array([5, 7], dtype=np.int32),
            "feature_vis_rgb_red": np.array([10, 20], dtype=np.uint8),
            "feature_vis_rgb_green": np.array([30, 40], dtype=np.uint8),
            "feature_vis_rgb_blue": np.array([50, 60], dtype=np.uint8),
            "bsr_candidate_score": np.array([0.25, 0.75], dtype=np.float32),
        },
        "metadata": {
            "dataset_name": "DummySet",
            "num_classes": 2,
            "class_names": ["ground", "car", "void"],
            "class_colors": [[1, 2, 3], [4, 5, 6], [0, 0, 0]],
            "stuff_classes": [0],
            "visualization_keys": ["bsr_candidate_score"],
            "export_fields": [
                "semantic_pred",
                "obj_pred",
                "feature_vis_rgb_red",
                "feature_vis_rgb_green",
                "feature_vis_rgb_blue",
                "bsr_candidate_score",
            ],
        },
    }
    out_path = tmp_path / "scene_pred.ply"

    batch_inference.save_predictions(str(out_path), save_data, "DummySet")

    with open(out_path, "rb") as f:
        header = f.read().split(b"end_header\n", 1)[0].decode("ascii")
    assert "property int semantic_pred" in header
    assert "property int obj_pred" in header
    assert "property uchar feature_vis_rgb_red" in header
    assert "property float bsr_candidate_score" in header

    meta = json.loads((tmp_path / "scene_pred.ply.meta.json").read_text(encoding="utf-8"))
    assert meta["dataset_name"] == "DummySet"
    assert "bsr_candidate_score" in meta["export_fields"]
