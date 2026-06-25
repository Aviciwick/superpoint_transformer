import types

import torch

from src.models.semantic import build_semantic_wandb_metadata, safe_count_parameters


class _Dataset:
    num_classes = 8
    class_names = ["road", "marking"]
    stuff_classes = [0, 1]

    def __len__(self):
        return 12


def test_build_semantic_wandb_metadata_includes_dataset_training_and_bsr_fields():
    datamodule = types.SimpleNamespace(
        train_dataset=_Dataset(),
        val_dataset=_Dataset(),
        test_dataset=_Dataset(),
        hparams=types.SimpleNamespace(
            xy_tiling=None,
            pc_tiling=4,
            voxel=0.05,
            dataloader=types.SimpleNamespace(batch_size=1, num_workers=8),
        ),
    )
    trainer = types.SimpleNamespace(
        max_epochs=800,
        check_val_every_n_epoch=10,
        precision=32,
    )
    model_hparams = types.SimpleNamespace(
        optimizer=types.SimpleNamespace(keywords={"lr": 0.01, "weight_decay": 0.0001}),
        scheduler=types.SimpleNamespace(
            func=types.SimpleNamespace(__name__="CosineAnnealingLRWithWarmup"),
            keywords={"num_warmup": 20},
        ),
        bsr={
            "enable": True,
            "selector": {"topk_ratio": 0.15},
            "refiner": {"variant": "mixture", "n_sample": 64, "n_subregions": 2},
            "propagation": {"mode": "slot_all_points", "metric_level": "point"},
        },
    )

    config, summary = build_semantic_wandb_metadata(
        trainer=trainer,
        datamodule=datamodule,
        model_hparams=model_hparams,
        num_classes=8,
        class_names=["road", "marking"],
        stuff_classes=[0, 1],
        bsr_enabled=True,
        parameter_counts=(123, 100),
        commit_hash="abc123",
    )

    assert config["run/dataset_class"] == "_Dataset"
    assert config["run/class_names"] == ["road", "marking"]
    assert config["run/train_items"] == 12
    assert config["run/max_epochs"] == 800
    assert config["run/check_val_every_n_epoch"] == 10
    assert config["run/xy_tiling"] is None
    assert config["run/pc_tiling"] == 4
    assert config["run/optimizer"] == "SimpleNamespace"
    assert config["run/lr"] == 0.01
    assert config["run/weight_decay"] == 0.0001
    assert config["run/scheduler"] == "CosineAnnealingLRWithWarmup"
    assert config["run/scheduler_warmup"] == 20
    assert config["run/bsr_enabled"] is True
    assert config["run/bsr_variant"] == "mixture"
    assert config["run/bsr_propagation_mode"] == "slot_all_points"
    assert config["run/bsr_metric_level"] == "point"
    assert config["run/num_parameters"] == 123
    assert config["run/num_trainable_parameters"] == 100
    assert config["run/commit_hash"] == "abc123"
    assert summary["run/train_items"] == 12
    assert summary["run/bsr_enabled"] == 1


def test_safe_count_parameters_ignores_uninitialized_lazy_parameters():
    module = torch.nn.Sequential(
        torch.nn.LazyLinear(4),
        torch.nn.Linear(4, 2),
    )

    total, trainable = safe_count_parameters(module.parameters())

    assert total == 10
    assert trainable == 10
