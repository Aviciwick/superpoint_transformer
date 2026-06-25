"""Batch inference for semantic and BSR checkpoints across supported datasets."""

import argparse
import bisect
import json
import os
import re
import sys
import time
from collections import OrderedDict
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any, Optional, Sequence

import hydra
import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import ConcatDataset
from tqdm import tqdm

sys.path.append(os.getcwd())

from src.data import InstanceData
from src.utils.hydra import init_config
from src.utils.color import feats_to_plotly_rgb
from src.visualization.visualization import visualize_3d


DEFAULT_VOID_NAME = "void"
DEFAULT_VOID_COLOR = [0, 0, 0]
DEFAULT_VIS_MAX_POINTS = 50000
FEATURE_VIS_FIELD = "feature_vis_rgb"
BSR_VIS_KEYS = [
    "bsr_candidate_mask",
    "bsr_candidate_score",
    "bsr_assignment_entropy",
    "bsr_secondary_slot_mass",
    "bsr_dual_slot_active",
    "bsr_slot_diversity",
    "bsr_point_semantic_pred",
    "bsr_point_propagation_mask",
    "bsr_point_propagation_coverage",
]
POINT_EXPORT_TENSOR_KEYS = [
    "super_sampling",
    "bsr_point_logits",
    "bsr_slot_affinity",
]


def _to_python_list(values: Optional[Sequence[Any]]) -> Optional[list]:
    if values is None:
        return None
    if isinstance(values, np.ndarray):
        return values.tolist()
    return list(values)


def _unwrap_dataset(dataset):
    current = dataset
    while hasattr(current, "datasets") and getattr(current, "datasets"):
        current = current.datasets[0]
    return current


def _normalize_label_metadata(class_names, class_colors, num_classes, stuff_classes):
    class_names = _to_python_list(class_names)
    class_colors = _to_python_list(class_colors)
    stuff_classes = [] if stuff_classes is None else sorted({int(x) for x in stuff_classes})

    if num_classes is None:
        if class_names is not None and len(class_names) > 0:
            num_classes = len(class_names) - 1
        elif class_colors is not None and len(class_colors) > 0:
            num_classes = len(class_colors) - 1
        else:
            raise ValueError("Could not infer dataset num_classes from dataset metadata")
    num_classes = int(num_classes)
    expected_len = num_classes + 1

    if class_names is None:
        class_names = [f"class_{i}" for i in range(num_classes)] + [DEFAULT_VOID_NAME]
    elif len(class_names) < expected_len:
        class_names = class_names + [f"class_{i}" for i in range(len(class_names), num_classes)]
        class_names.append(DEFAULT_VOID_NAME)
    elif len(class_names) == num_classes:
        class_names = class_names + [DEFAULT_VOID_NAME]

    if class_colors is not None:
        class_colors = [list(map(int, color)) for color in class_colors]
        if len(class_colors) < expected_len:
            class_colors = class_colors + [DEFAULT_VOID_COLOR for _ in range(expected_len - len(class_colors))]
        elif len(class_colors) == num_classes:
            class_colors = class_colors + [DEFAULT_VOID_COLOR]

    stuff_classes = [x for x in stuff_classes if 0 <= x < num_classes]
    return class_names, class_colors, num_classes, stuff_classes


def resolve_dataset_metadata(dataset):
    base_dataset = _unwrap_dataset(dataset)
    class_names = getattr(base_dataset, "class_names", None)
    class_colors = getattr(base_dataset, "class_colors", None)
    num_classes = getattr(base_dataset, "num_classes", None)
    stuff_classes = getattr(base_dataset, "stuff_classes", None)

    class_names, class_colors, num_classes, stuff_classes = _normalize_label_metadata(
        class_names=class_names,
        class_colors=class_colors,
        num_classes=num_classes,
        stuff_classes=stuff_classes,
    )

    return {
        "dataset_name": base_dataset.__class__.__name__,
        "class_names": class_names,
        "class_colors": class_colors,
        "num_classes": num_classes,
        "stuff_classes": stuff_classes,
    }


def _read_checkpoint_experiment_hint(ckpt_path: str) -> Optional[str]:
    run_dir = _checkpoint_run_dir(ckpt_path)
    overrides_path = os.path.join(run_dir, ".hydra", "overrides.yaml")
    if not os.path.isfile(overrides_path):
        return None

    with open(overrides_path, "r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if line.startswith("- "):
                line = line[2:]
            if line.startswith("experiment="):
                return line.split("=", 1)[1].strip()
    return None


def _checkpoint_run_dir(ckpt_path: str) -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(ckpt_path)))


def _checkpoint_hydra_config_path(ckpt_path: str) -> str:
    return os.path.join(_checkpoint_run_dir(ckpt_path), ".hydra", "config.yaml")


def _register_eval_resolver():
    if not OmegaConf.has_resolver("eval"):
        OmegaConf.register_new_resolver("eval", eval)


def _inference_override_dotlist(ckpt_path: str) -> list:
    return [
        "datamodule.dataloader.batch_size=1",
        "datamodule.dataloader.num_workers=0",
        f"ckpt_path={ckpt_path}",
    ]


def _load_inference_config(experiment: str, ckpt_path: str):
    checkpoint_config_path = _checkpoint_hydra_config_path(ckpt_path)
    inference_overrides = _inference_override_dotlist(ckpt_path)

    if os.path.isfile(checkpoint_config_path):
        _register_eval_resolver()
        cfg = OmegaConf.load(checkpoint_config_path)
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(inference_overrides))
        return cfg, checkpoint_config_path

    overrides = [f"experiment={experiment}", *inference_overrides]
    return init_config(config_name="train.yaml", overrides=overrides), "current experiment config"


def _extract_checkpoint_signature(checkpoint: dict, ckpt_path: str) -> dict:
    hyper_parameters = checkpoint.get("hyper_parameters") or {}
    bsr_cfg = hyper_parameters.get("bsr") or {}
    return {
        "experiment_hint": _read_checkpoint_experiment_hint(ckpt_path),
        "num_classes": hyper_parameters.get("num_classes"),
        "down_dim": _to_python_list(hyper_parameters.get("_down_dim")),
        "up_dim": _to_python_list(hyper_parameters.get("_up_dim")),
        "point_out_dim": hyper_parameters.get("_point_out_dim"),
        "point_hf_dim": hyper_parameters.get("_point_hf_dim"),
        "bsr_enable": bool(bsr_cfg.get("enable", False)),
    }


def _extract_model_signature(cfg, model) -> dict:
    return {
        "experiment_hint": None,
        "num_classes": int(getattr(model, "num_classes", cfg.model.num_classes)),
        "down_dim": _to_python_list(cfg.model._down_dim),
        "up_dim": _to_python_list(cfg.model._up_dim),
        "point_out_dim": getattr(model, "point_out_dim", cfg.model._point_out_dim),
        "point_hf_dim": cfg.model._point_hf_dim,
        "bsr_enable": bool(cfg.model.bsr.enable),
    }


def _format_signature(signature: dict) -> str:
    return (
        f"num_classes={signature.get('num_classes')}, "
        f"down_dim={signature.get('down_dim')}, "
        f"up_dim={signature.get('up_dim')}, "
        f"point_out_dim={signature.get('point_out_dim')}, "
        f"point_hf_dim={signature.get('point_hf_dim')}, "
        f"bsr_enable={signature.get('bsr_enable')}"
    )


def _validate_checkpoint_compatibility(
        *,
        requested_experiment: str,
        checkpoint_signature: dict,
        model_signature: dict):
    mismatches = []

    for key, label in (
            ("num_classes", "num_classes"),
            ("down_dim", "down_dim"),
            ("up_dim", "up_dim"),
            ("point_out_dim", "point_out_dim"),
            ("point_hf_dim", "point_hf_dim"),
            ("bsr_enable", "bsr_enable")):
        ckpt_value = checkpoint_signature.get(key)
        model_value = model_signature.get(key)
        if ckpt_value is None or model_value is None:
            continue
        if ckpt_value != model_value:
            mismatches.append(f"{label}: checkpoint={ckpt_value}, current={model_value}")

    experiment_hint = checkpoint_signature.get("experiment_hint")
    if experiment_hint and experiment_hint != requested_experiment:
        mismatches.insert(0, f"experiment: checkpoint={experiment_hint}, requested={requested_experiment}")

    if not mismatches:
        return

    message_lines = [
        "Checkpoint/config mismatch detected before loading weights.",
        f"Requested experiment: {requested_experiment}",
    ]
    if experiment_hint:
        message_lines.append(f"Checkpoint was trained with: {experiment_hint}")
    message_lines.append(f"Checkpoint signature: {_format_signature(checkpoint_signature)}")
    message_lines.append(f"Current config signature: {_format_signature(model_signature)}")
    message_lines.append("Mismatches:")
    message_lines.extend(f"  - {item}" for item in mismatches)
    if experiment_hint and experiment_hint != requested_experiment:
        message_lines.append(
            f"Use `--experiment {experiment_hint}` with this checkpoint, "
            f"or provide a checkpoint trained for `{requested_experiment}`."
        )
    raise ValueError("\n".join(message_lines))


def _resolve_cloud_ids(dataset):
    if hasattr(dataset, "cloud_ids"):
        return list(dataset.cloud_ids)
    if hasattr(dataset, "datasets") and getattr(dataset, "datasets"):
        cloud_ids = []
        for sub_dataset in dataset.datasets:
            sub_ids = _resolve_cloud_ids(sub_dataset)
            if sub_ids is None:
                return None
            cloud_ids.extend(sub_ids)
        return cloud_ids
    return None


def _select_dataset(datamodule, stage: str):
    if stage == "train":
        return datamodule.train_dataset
    if stage == "val":
        return datamodule.val_dataset
    if stage == "trainval":
        return ConcatDataset([datamodule.train_dataset, datamodule.val_dataset])
    return datamodule.test_dataset


def _select_inference_transforms(datamodule, stage: str):
    if stage == "test":
        transform = datamodule.test_transform
        on_device_transform = getattr(datamodule, "on_device_test_transform", None)
        if on_device_transform is None:
            on_device_transform = getattr(datamodule, "on_device_val_transform", None)
        return transform, on_device_transform

    transform = datamodule.val_transform
    if transform is None:
        transform = datamodule.test_transform
    on_device_transform = getattr(datamodule, "on_device_val_transform", None)
    if on_device_transform is None:
        on_device_transform = getattr(datamodule, "on_device_test_transform", None)
    return transform, on_device_transform


def _build_inference_dataset(datamodule, stage: str):
    transform, on_device_transform = _select_inference_transforms(datamodule, stage)
    return datamodule.dataset_class(
        datamodule.hparams.data_dir,
        stage=stage,
        transform=transform,
        pre_transform=datamodule.pre_transform,
        on_device_transform=on_device_transform,
        **datamodule.kwargs,
    )


def _resolve_dataset_sample(dataset, idx: int):
    if isinstance(dataset, ConcatDataset):
        if idx < 0:
            if -idx > len(dataset):
                raise ValueError("absolute value of index should not exceed dataset length")
            idx = len(dataset) + idx
        sub_dataset_idx = bisect.bisect_right(dataset.cumulative_sizes, idx)
        sample_idx = idx
        if sub_dataset_idx > 0:
            sample_idx -= dataset.cumulative_sizes[sub_dataset_idx - 1]
        sub_dataset = dataset.datasets[sub_dataset_idx]
        sample = sub_dataset[sample_idx]
        return sample, getattr(sub_dataset, "on_device_transform", None)

    return dataset[idx], getattr(dataset, "on_device_transform", None)


def _ensure_inference_point_keys(cfg):
    datamodule_cfg = cfg.datamodule
    point_load_keys = list(getattr(datamodule_cfg, "point_load_keys", []) or [])
    if "rgb" not in point_load_keys:
        point_load_keys.append("rgb")
    datamodule_cfg.point_load_keys = point_load_keys


def _collect_visualization_keys(nag):
    keys = []
    for key in BSR_VIS_KEYS:
        if hasattr(nag[0], key) and getattr(nag[0], key) is not None:
            keys.append(key)
    return keys


def _sanitize_property_name(name: str) -> str:
    name = re.sub(r"[^0-9A-Za-z_]+", "_", str(name).strip())
    name = re.sub(r"_+", "_", name).strip("_").lower()
    if not name:
        name = "field"
    if name[0].isdigit():
        name = f"field_{name}"
    return name


def _to_numpy_array(value):
    if value is None:
        return None
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    if isinstance(value, np.ndarray):
        return value
    return None


def _normalize_pos_offset(pos_offset, *, scene_id: str = "scene") -> Optional[np.ndarray]:
    array = _to_numpy_array(pos_offset)
    if array is None:
        return None
    array = np.asarray(array)
    if array.size == 0:
        return None
    if array.shape == (3,):
        return array
    if array.ndim == 2 and array.shape[1] == 3:
        offsets = array
    elif array.ndim == 1 and array.size % 3 == 0:
        offsets = array.reshape(-1, 3)
    else:
        print(
            f"Warning: {scene_id} has unsupported pos_offset shape {array.shape}; "
            "coordinates will be exported without restoring the global offset."
        )
        return None
    if offsets.shape[0] == 1:
        return offsets[0]
    if np.allclose(offsets, offsets[0:1]):
        return offsets[0]
    print(
        f"Warning: {scene_id} has multiple different pos_offset values with "
        f"shape {array.shape}; coordinates will be exported without restoring "
        "the global offset."
    )
    return None


def _extract_dense_instance_ids(value, *, num_points: int, num_classes: int) -> Optional[np.ndarray]:
    if value is None:
        return None
    if isinstance(value, InstanceData):
        try:
            obj, _, _ = value.major(num_classes=num_classes)
            return obj.detach().cpu().numpy().astype(np.int32, copy=False)
        except Exception as exc:
            print(f"Warning: failed to densify InstanceData: {exc}")
            return None

    array = _to_numpy_array(value)
    if array is None:
        return None
    if array.ndim == 2 and array.shape[1] == 1:
        array = array[:, 0]
    if array.ndim != 1 or array.shape[0] != num_points:
        return None
    return array.astype(np.int32, copy=False)


def _append_export_field(export_fields: OrderedDict, name: str, value, *, num_points: int):
    array = _to_numpy_array(value)
    if array is None:
        return
    if array.ndim == 0:
        return
    if array.shape[0] != num_points:
        return
    if array.ndim == 1:
        export_fields[name] = array
        return
    if array.ndim == 2:
        export_fields[name] = array


def _flatten_export_fields(export_fields: OrderedDict, *, num_points: int) -> OrderedDict:
    flat_fields = OrderedDict()
    for name, value in export_fields.items():
        array = _to_numpy_array(value)
        if array is None or array.ndim == 0 or array.shape[0] != num_points:
            continue
        sanitized = _sanitize_property_name(name)
        if array.ndim == 1:
            flat_fields[sanitized] = array
            continue
        if array.ndim != 2:
            continue
        if sanitized.endswith("_rgb") and array.shape[1] == 3:
            channel_names = ("red", "green", "blue")
            for channel_idx, channel_name in enumerate(channel_names):
                flat_fields[f"{sanitized}_{channel_name}"] = array[:, channel_idx]
            continue
        for channel_idx in range(array.shape[1]):
            flat_fields[f"{sanitized}_{channel_idx}"] = array[:, channel_idx]
    return flat_fields


def _field_storage_spec(name: str, values: np.ndarray):
    if np.issubdtype(values.dtype, np.floating):
        return values.astype(np.float32, copy=False), "float", "<f4"
    if np.issubdtype(values.dtype, np.bool_):
        return values.astype(np.uint8, copy=False), "uchar", "u1"
    if np.issubdtype(values.dtype, np.integer):
        min_value = int(values.min()) if values.size else 0
        max_value = int(values.max()) if values.size else 0
        if min_value >= 0 and max_value <= 255 and name.endswith(("_red", "_green", "_blue")):
            return values.astype(np.uint8, copy=False), "uchar", "u1"
        return values.astype(np.int32, copy=False), "int", "<i4"
    return values.astype(np.float32, copy=False), "float", "<f4"


def _prepare_save_data(nag, metadata: dict):
    """
    从 NAG 对象中提取并准备需要保存的数据（numpy 数组）。
    在 GPU 推理完成后调用，将所有 tensor 移到 CPU 并转为 numpy。

    Args:
        nag: 包含预测结果的 NAG 对象

    Returns:
        dict: 包含 pos, rgb, pred, obj_pred, sp_indices 等推理结果的字典
    """
    data = nag[0]
    pos = data.pos.cpu().numpy()

    if hasattr(data, 'pos_offset') and data.pos_offset is not None:
        pos_offset = _normalize_pos_offset(data.pos_offset, scene_id=metadata.get("dataset_name", "scene"))
        if pos_offset is not None:
            pos = pos + pos_offset

    if hasattr(data, 'rgb') and data.rgb is not None:
        rgb = data.rgb.cpu().numpy()
        if rgb.max() <= 1.0:
            rgb = rgb * 255
    else:
        rgb = np.zeros_like(pos)

    num_points = pos.shape[0]
    num_classes = int(metadata["num_classes"])

    if data.semantic_pred is not None:
        pred = data.semantic_pred.detach().cpu().numpy()
        if pred.ndim > 1:
            pred = np.argmax(pred, axis=1)
        pred = pred.astype(np.int32, copy=False)
    else:
        pred = np.zeros(num_points, dtype=np.int32)

    export_fields = OrderedDict()
    export_fields["semantic_pred"] = pred

    obj_pred = _extract_dense_instance_ids(
        getattr(data, "obj_pred", None),
        num_points=num_points,
        num_classes=num_classes,
    )
    if obj_pred is None:
        obj_pred = np.full(num_points, -1, dtype=np.int32)
    export_fields["obj_pred"] = obj_pred

    obj_gt = _extract_dense_instance_ids(
        getattr(data, "obj", None),
        num_points=num_points,
        num_classes=num_classes,
    )
    if obj_gt is not None:
        export_fields["obj"] = obj_gt

    sp_indices = []
    if hasattr(nag, 'num_levels'):
        num_levels = nag.num_levels
        current_sp = None
        for i in range(num_levels - 1):
            if hasattr(nag[i], 'super_index') and nag[i].super_index is not None:
                if i == 0:
                    current_sp = nag[i].super_index.cpu().numpy()
                else:
                    current_sp = nag[i].super_index.cpu().numpy()[current_sp]
                sp_indices.append(current_sp)

    has_gt = False
    gt_label = np.full(pos.shape[0], -1, dtype=np.int32)
    if hasattr(data, 'y') and data.y is not None:
        gt_tensor = data.y.cpu().numpy()
        if gt_tensor.ndim > 1:
            gt_label = np.argmax(gt_tensor, axis=1)
        else:
            gt_label = gt_tensor
        gt_label = gt_label.astype(np.int32, copy=False)
        has_gt = True

    is_error = np.zeros(pos.shape[0], dtype=np.int32)
    if has_gt:
        is_error = (pred != gt_label).astype(np.int32)
        export_fields["gt_label"] = gt_label
        export_fields["is_error"] = is_error

    for i, sp in enumerate(sp_indices, start=1):
        export_fields[f"sp_level_{i}"] = np.asarray(sp, dtype=np.int32)

    if getattr(data, "x", None) is not None and torch.is_tensor(data.x) and data.x.shape[0] == num_points:
        export_fields[FEATURE_VIS_FIELD] = feats_to_plotly_rgb(
            data.x.detach().cpu(),
            normalize=True,
            colorscale=None,
        )

    for key in POINT_EXPORT_TENSOR_KEYS + _collect_visualization_keys(nag):
        if key in export_fields:
            continue
        _append_export_field(
            export_fields,
            key,
            getattr(data, key, None),
            num_points=num_points,
        )

    flat_export_fields = _flatten_export_fields(export_fields, num_points=num_points)
    return {
        'pos': pos,
        'rgb': rgb,
        'num_points': num_points,
        'export_fields': flat_export_fields,
        'metadata': {
            'dataset_name': metadata["dataset_name"],
            'num_classes': metadata["num_classes"],
            'class_names': metadata["class_names"],
            'class_colors': metadata["class_colors"],
            'stuff_classes': metadata["stuff_classes"],
            'visualization_keys': _collect_visualization_keys(nag),
            'export_fields': list(flat_export_fields.keys()),
        }
    }


def save_predictions(path: str, save_data: dict, dataset_name: str = None):
    """
    保存预测结果到文件。使用二进制 PLY 格式代替 ASCII 以大幅提升写入速度。
    对于 152 万点的场景，二进制写入比 ASCII 快 10-100 倍。

    Args:
        path: 输出文件路径
        save_data: 由 _prepare_save_data 返回的字典
        dataset_name: 数据集名称（用于格式调整）
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)

    pos = save_data['pos']
    rgb = save_data['rgb']
    num_points = save_data['num_points']
    export_fields = save_data['export_fields']
    metadata = dict(save_data.get('metadata', {}))
    metadata['dataset_name'] = metadata.get('dataset_name', dataset_name)
    metadata['point_properties'] = ['x', 'y', 'z', 'red', 'green', 'blue'] + list(export_fields.keys())
    metadata['num_point_properties'] = len(metadata['point_properties'])

    ext = os.path.splitext(path)[1].lower()

    if ext == '.ply':
        _save_binary_ply(path, pos, rgb, export_fields, num_points)
    else:
        column_names = ['x', 'y', 'z', 'red', 'green', 'blue'] + list(export_fields.keys())
        columns = [pos, np.clip(rgb, 0, 255).astype(np.uint8)]
        fmt_list = ['%.6f'] * 3 + ['%d'] * 3
        for values in export_fields.values():
            column = np.asarray(values).reshape(num_points, 1)
            columns.append(column)
            if np.issubdtype(column.dtype, np.floating):
                fmt_list.append('%.6f')
            else:
                fmt_list.append('%d')
        output_data = np.column_stack(columns)
        np.savetxt(
            path,
            output_data,
            fmt=' '.join(fmt_list),
            header=' '.join(column_names),
            comments='# ',
        )

    metadata_path = f"{path}.meta.json"
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print(f"Saved: {path}")


def _save_binary_ply(path, pos, rgb, export_fields, num_points):
    """
    以二进制 little-endian 格式保存 PLY 文件。
    相比 ASCII 格式，写入速度提升 10-100 倍，文件体积缩小约 75%。

    Args:
        path: 输出文件路径
        pos: 点坐标 (N, 3) float
        rgb: 颜色 (N, 3) uint8
        pred: 语义预测 (N,) int
        obj_pred: 实例预测 (N,) int
        sp_indices: 超点索引列表
        has_gt: 是否有地面真值标签
        gt_label: 地面真值标签 (N,) int
        is_error: 预测是否错误的标记 (N,) int
        num_points: 点数
    """
    # 构建 PLY header
    header_lines = [
        "ply",
        "format binary_little_endian 1.0",
        f"element vertex {num_points}",
        "property float x",
        "property float y",
        "property float z",
        "property uchar red",
        "property uchar green",
        "property uchar blue",
    ]

    normalized_fields = OrderedDict()
    for name, values in export_fields.items():
        field_values, ply_type, dtype_str = _field_storage_spec(name, np.asarray(values))
        normalized_fields[name] = (field_values, ply_type, dtype_str)
        header_lines.append(f"property {ply_type} {name}")
    header_lines.append("end_header")
    header = "\n".join(header_lines) + "\n"

    dt_fields = [
        ('x', '<f4'), ('y', '<f4'), ('z', '<f4'),
        ('r', 'u1'), ('g', 'u1'), ('b', 'u1'),
    ]
    for name, (_, _, dtype_str) in normalized_fields.items():
        dt_fields.append((name, dtype_str))

    vertex_dtype = np.dtype(dt_fields)
    vertices = np.empty(num_points, dtype=vertex_dtype)

    # 向量化赋值（无 Python 循环）
    vertices['x'] = pos[:, 0].astype(np.float32)
    vertices['y'] = pos[:, 1].astype(np.float32)
    vertices['z'] = pos[:, 2].astype(np.float32)
    rgb_u8 = np.clip(rgb, 0, 255).astype(np.uint8)
    vertices['r'] = rgb_u8[:, 0]
    vertices['g'] = rgb_u8[:, 1]
    vertices['b'] = rgb_u8[:, 2]
    for name, (values, _, _) in normalized_fields.items():
        vertices[name] = values

    with open(path, 'wb') as f:
        f.write(header.encode('ascii'))
        f.write(vertices.tobytes())


def generate_visualization(
        nag,
        output_html: str,
        *,
        class_names,
        class_colors,
        stuff_classes,
        num_classes,
        max_points,
        centroids,
        h_edge,
        h_edge_width):
    """
    生成可视化 HTML 文件。
    
    Args:
        nag: NAG 对象
        output_html: 输出 HTML 路径
        class_names: 类别名称列表
        class_colors: 类别颜色列表
        stuff_classes: stuff 类别索引列表
        num_classes: 类别数量
    """
    nag = nag.to('cpu')
    extra_keys = _collect_visualization_keys(nag)

    vis_output = visualize_3d(
        nag,
        keys=extra_keys,
        class_names=class_names,
        class_colors=class_colors,
        stuff_classes=stuff_classes,
        num_classes=num_classes,
        max_points=max_points,
        centroids=centroids,
        h_edge=centroids and h_edge,
        h_edge_width=h_edge_width,
    )

    fig = vis_output['figure']
    fig.update_layout(
        autosize=True,
        width=None,
        height=None,
        margin=dict(l=0, r=0, b=0, t=0)
    )
    
    fig.write_html(output_html, config={'responsive': True})
    print(f"Visualization saved: {output_html}")


def run_inference_on_batch(model, batch, device):
    """
    对单个 batch 执行推理。
    
    Args:
        model: 模型
        batch: 输入数据 batch
        device: 计算设备
    
    Returns:
        NAG: 包含预测结果的 NAG 对象
    """
    batch = batch.to(device)
    
    # Ensure super_index and other indices are long to prevent torch_scatter from crashing
    if hasattr(batch, 'num_levels'):
        for i in range(getattr(batch, 'num_levels')):
            if hasattr(batch[i], 'super_index') and batch[i].super_index is not None:
                batch[i].super_index = batch[i].super_index.long()
            if hasattr(batch[i], 'edge_index') and batch[i].edge_index is not None:
                batch[i].edge_index = batch[i].edge_index.long()

    with torch.no_grad():
        output = model(batch)

    if not output.multi_stage:
        logits = output.logits
        pred = torch.argmax(logits, dim=1)
        batch[1].semantic_pred = pred
        batch[1].logits = logits
        if getattr(output, "point_logits", None) is not None:
            batch[0].logits = output.point_logits
            batch[0].semantic_pred = torch.argmax(output.point_logits, dim=1)
        else:
            batch[0].semantic_pred = pred[batch[0].super_index]
            batch[0].logits = logits[batch[0].super_index]
    else:
        for i, logits in enumerate(output.logits):
            pred = torch.argmax(logits, dim=1)
            batch[i + 1].semantic_pred = pred
            batch[i + 1].logits = logits
            if i == 0:
                if getattr(output, "point_logits", None) is not None:
                    batch[0].logits = output.point_logits
                    batch[0].semantic_pred = torch.argmax(output.point_logits, dim=1)
                else:
                    batch[0].semantic_pred = pred[batch[0].super_index]
                    batch[0].logits = logits[batch[0].super_index]

    if hasattr(output, 'bsr_output') and output.bsr_output is not None \
            and hasattr(model, '_attach_bsr_tracking_metadata'):
        model._attach_bsr_tracking_metadata(batch, output.bsr_output)

    try:
        res = output.voxel_panoptic_pred(super_index=batch[0].super_index)
        if isinstance(res, tuple) and len(res) == 3:
            _, _, vox_obj_pred = res
            batch[0].obj_pred = vox_obj_pred
    except Exception:
        pass

    return batch


def main():
    parser = argparse.ArgumentParser(description="Batch inference on entire dataset")
    parser.add_argument('--experiment', type=str, required=True,
                        help='Experiment config name (e.g., semantic/dales_bsr, semantic/kitti360_bsr)')
    parser.add_argument('--ckpt', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--output_dir', type=str, default='output/batch_predictions',
                        help='Output directory for predictions')
    parser.add_argument('--stage', type=str, default='test', 
                        choices=['train', 'val', 'test', 'trainval'],
                        help='Dataset stage to run inference on')
    parser.add_argument('--visualize', action='store_true',
                        help='Generate visualization HTML for each scene')
    parser.add_argument('--output_format', type=str, default='ply',
                        choices=['txt', 'ply'],
                        help='Output file format')
    parser.add_argument('--limit', type=int, default=None,
                        help='Limit number of scenes to process (for testing)')
    parser.add_argument('--scene_ids', type=str, nargs='+', default=None,
                        help='Specific scene IDs to process (e.g., Area_5/office_1)')
    parser.add_argument('--visualize_max_points', type=int, default=DEFAULT_VIS_MAX_POINTS,
                        help='Maximum number of points shown in each visualization')
    parser.add_argument('--visualize_centroids', action='store_true',
                        help='Show superpoint centroids in the HTML visualization')
    parser.add_argument('--visualize_h_edge', action='store_true',
                        help='Show horizontal superedges in the HTML visualization')
    parser.add_argument('--visualize_h_edge_width', type=float, default=1.5,
                        help='Line width for horizontal superedges in the HTML visualization')
    args = parser.parse_args()
    
    os.environ["PROJECT_ROOT"] = os.path.dirname(os.path.abspath(__file__))
    
    print(f"Experiment: {args.experiment}")
    print(f"Checkpoint: {args.ckpt}")
    print(f"Output directory: {args.output_dir}")
    print(f"Stage: {args.stage}")
    
    print(f"Loading checkpoint from {args.ckpt}...")
    checkpoint = torch.load(args.ckpt, map_location='cpu')
    has_legacy_hspt = any('hspt.' in k for k in checkpoint['state_dict'].keys())
    checkpoint_signature = _extract_checkpoint_signature(checkpoint, args.ckpt)
    
    print("Loading configuration...")
    if has_legacy_hspt:
        print("Detected legacy H-SPT weights in the checkpoint. They will be ignored by the current semantic inference path.")
        
    cfg, config_source = _load_inference_config(
        experiment=args.experiment,
        ckpt_path=args.ckpt,
    )
    print(f"Configuration source: {config_source}")
    _ensure_inference_point_keys(cfg)
    
    print("Instantiating datamodule...")
    datamodule = hydra.utils.instantiate(cfg.datamodule)

    dataset = _build_inference_dataset(datamodule, args.stage)
    metadata = resolve_dataset_metadata(dataset)

    print(
        "Resolved dataset metadata:",
        metadata["dataset_name"],
        f"num_classes={metadata['num_classes']}",
        f"stuff_classes={metadata['stuff_classes']}",
    )
    if hasattr(dataset, "stage"):
        print(f"Resolved dataset stage: {dataset.stage}")

    print("Instantiating model...")
    model = hydra.utils.instantiate(cfg.model)
    model_signature = _extract_model_signature(cfg, model)
    _validate_checkpoint_compatibility(
        requested_experiment=args.experiment,
        checkpoint_signature=checkpoint_signature,
        model_signature=model_signature,
    )

    model.load_state_dict(checkpoint['state_dict'], strict=False)
    if hasattr(model, "bsr_store_slot_affinity"):
        model.bsr_store_slot_affinity = True
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model = model.eval().to(device)
    
    if hasattr(model, 'net'):
        model.net.store_features = True

    cloud_ids = _resolve_cloud_ids(dataset)
    
    if args.scene_ids:
        if cloud_ids is not None:
            scene_id_to_idx = {cid: idx for idx, cid in enumerate(cloud_ids)}
            valid_scene_ids = [sid for sid in args.scene_ids if sid in scene_id_to_idx]
            if not valid_scene_ids:
                print(f"Warning: None of the specified scene IDs found.")
                print(f"Available IDs (first 10): {cloud_ids[:10]}...")
            scene_indices = [scene_id_to_idx[sid] for sid in valid_scene_ids]
        else:
            scene_indices = [int(sid) for sid in args.scene_ids if sid.isdigit()]
    else:
        scene_indices = list(range(len(dataset)))
    
    if args.limit:
        scene_indices = scene_indices[:args.limit]
    
    print(f"Processing {len(scene_indices)} scenes...")
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    results_summary = []
    
    # --- 优化: 使用线程池异步保存/可视化，以及后台预加载 ---
    io_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix='io_worker')
    io_futures: list[Future] = []
    
    def _preload_scene(ds, scene_idx):
        """
        在后台线程中预加载下一个场景的数据。
        
        Args:
            ds: 数据集对象
            scene_idx: 场景索引
        
        Returns:
            预加载的 batch 数据及其对应的 on-device transform
        """
        return _resolve_dataset_sample(ds, scene_idx)
    
    preload_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='preload')
    prefetch_future: Future = None
    
    total_infer_time = 0.0
    
    for i, idx in enumerate(tqdm(scene_indices, desc="Processing scenes")):
        try:
            # --- 获取数据: 优先使用预加载的结果 ---
            t_load_start = time.perf_counter()
            if prefetch_future is not None:
                batch, on_device_transform = prefetch_future.result()
            else:
                batch, on_device_transform = _resolve_dataset_sample(dataset, idx)
            t_load_end = time.perf_counter()
            
            # --- 提交下一个场景的预加载 ---
            if i + 1 < len(scene_indices):
                next_idx = scene_indices[i + 1]
                prefetch_future = preload_executor.submit(_preload_scene, dataset, next_idx)
            else:
                prefetch_future = None
            
            # NAG 对象的点数据在 batch[0].pos 中，而不是 batch.pos
            if hasattr(batch, 'num_levels'):
                if not hasattr(batch[0], 'pos') or batch[0].pos is None or batch[0].pos.shape[0] == 0:
                    print(f"Skipping empty scene at index {idx}")
                    continue
            else:
                if not hasattr(batch, 'pos') or batch.pos is None or batch.pos.shape[0] == 0:
                    print(f"Skipping empty scene at index {idx}")
                    continue
            
            scene_id = None
            if cloud_ids is not None and idx < len(cloud_ids):
                scene_id = cloud_ids[idx]
            elif hasattr(batch, 'cloud_id'):
                scene_id = batch.cloud_id
            elif hasattr(batch, 'id'):
                scene_id = batch.id
            else:
                scene_id = f"scene_{idx}"
            scene_id = str(scene_id)
            scene_id_safe = scene_id.replace('/', '_').replace('\\', '_')
            
            # --- GPU 推理 ---
            if on_device_transform:
                batch = on_device_transform(batch)
            
            t_infer_start = time.perf_counter()
            batch = run_inference_on_batch(model, batch, device)
            if device.type == "cuda":
                torch.cuda.synchronize()  # 确保 GPU 推理完成后再计时
            t_infer_end = time.perf_counter()
            total_infer_time += (t_infer_end - t_infer_start)
            
            # --- 在提交IO前，先在主线程中提取数据到 CPU numpy ---
            num_points = batch[0].pos.shape[0]
            save_data = _prepare_save_data(batch, metadata)
            
            # --- 异步提交保存和可视化到线程池 ---
            output_file = os.path.join(args.output_dir, f"{scene_id_safe}_pred.{args.output_format}")
            io_futures.append(
                io_executor.submit(save_predictions, output_file, save_data, args.experiment)
            )
            
            if args.visualize:
                output_html = os.path.join(args.output_dir, f"{scene_id_safe}_vis.html")
                batch_cpu = batch.to('cpu')
                io_futures.append(
                    io_executor.submit(
                        generate_visualization,
                        batch_cpu,
                        output_html,
                        class_names=metadata["class_names"],
                        class_colors=metadata["class_colors"],
                        stuff_classes=metadata["stuff_classes"],
                        num_classes=metadata["num_classes"],
                        max_points=args.visualize_max_points,
                        centroids=args.visualize_centroids,
                        h_edge=args.visualize_h_edge,
                        h_edge_width=args.visualize_h_edge_width,
                    )
                )
            
            results_summary.append({
                'scene_id': scene_id,
                'num_points': num_points,
                'output_file': output_file
            })
            
            print(f"  [{scene_id}] load={t_load_end - t_load_start:.1f}s, "
                  f"infer={t_infer_end - t_infer_start:.1f}s, points={num_points}")
            
        except Exception as e:
            print(f"Error processing scene {idx}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # --- 等待所有后台 IO 任务完成 ---
    print("Waiting for async IO tasks to complete...")
    for f in io_futures:
        try:
            f.result()
        except Exception as e:
            print(f"IO task error: {e}")
    
    io_executor.shutdown(wait=True)
    preload_executor.shutdown(wait=True)
    
    summary_file = os.path.join(args.output_dir, "inference_summary.txt")
    with open(summary_file, 'w') as f:
        f.write(f"Experiment: {args.experiment}\n")
        f.write(f"Checkpoint: {args.ckpt}\n")
        f.write(f"Stage: {args.stage}\n")
        f.write(f"Total scenes processed: {len(results_summary)}\n")
        f.write(f"Total inference time: {total_infer_time:.2f}s\n")
        f.write("-" * 50 + "\n")
        for result in results_summary:
            f.write(f"{result['scene_id']}: {result['num_points']} points -> {result['output_file']}\n")
    
    print(f"\nInference complete!")
    print(f"Total scenes processed: {len(results_summary)}")
    print(f"Total GPU inference time: {total_infer_time:.2f}s")
    print(f"Results saved to: {args.output_dir}")
    print(f"Summary saved to: {summary_file}")


if __name__ == "__main__":
    main()
