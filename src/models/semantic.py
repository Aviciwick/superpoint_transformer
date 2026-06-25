import torch
import os
import os.path as osp
from torch.nn import ModuleDict, ModuleList
from torch.nn.parameter import UninitializedParameter
import logging
from copy import deepcopy
from typing import Iterable, List, Tuple, Dict, Union, Any
from pytorch_lightning import LightningModule
from torchmetrics import MaxMetric, SumMetric, CatMetric
from pytorch_lightning.loggers.wandb import WandbLogger

from src.metrics import ConfusionMatrix
from src.utils import (
    loss_with_target_histogram,
    atomic_to_histogram,
    init_weights,
    wandb_confusion_matrix,
    knn_2,
    garbage_collection_cuda,
    SemanticSegmentationOutput,
    PartitionOutput,
    get_commit_hash)
from src.utils.torchmetrics import SafeMeanMetric
from src.nn import Classifier
from src.loss import MultiLoss
from src.optim.lr_scheduler import ON_PLATEAU_SCHEDULERS
from src.data import NAG, Data
from src.transforms import Transform, NAGSaveNodeIndex, PretrainedCNN

MeanMetric = SafeMeanMetric


# BSR-SPT 模块导入
try:
    from src.bsr import (
        BSRModule,
        build_candidate_point_cloud as build_bsr_candidate_point_cloud,
        build_packed_points as build_bsr_packed_points,
        compute_bsr_losses,
    )
    from src.bsr.geometry import extract_selector_handcrafted_features
    BSR_AVAILABLE = True
except ImportError:
    BSR_AVAILABLE = False

log = logging.getLogger(__name__)


__all__ = [
    'SemanticSegmentationModule',
    'build_semantic_wandb_metadata',
    'safe_count_parameters',
]


def _cfg_get(obj, key, default=None):
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    getter = getattr(obj, 'get', None)
    if getter is not None:
        try:
            return getter(key, default)
        except Exception:
            pass
    return getattr(obj, key, default)


def _cfg_get_nested(obj, keys, default=None):
    current = obj
    for key in keys:
        current = _cfg_get(current, key, default=None)
        if current is None:
            return default
    return current


def _safe_len(obj):
    if obj is None:
        return None
    try:
        return len(obj)
    except Exception:
        return None


def _callable_name(obj):
    if obj is None:
        return None
    func = getattr(obj, 'func', None)
    if func is not None:
        return getattr(func, '__name__', func.__class__.__name__)
    target = _cfg_get(obj, '_target_', None)
    if target is not None:
        return str(target).split('.')[-1]
    return obj.__class__.__name__


def _wandb_safe_value(value):
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if torch.is_tensor(value):
        value = value.detach().cpu()
        if value.numel() == 1:
            return value.item()
        return value.tolist()
    if isinstance(value, tuple):
        return [_wandb_safe_value(v) for v in value]
    if isinstance(value, list):
        return [_wandb_safe_value(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _wandb_safe_value(v) for k, v in value.items()}
    items = getattr(value, 'items', None)
    if items is not None:
        try:
            return {str(k): _wandb_safe_value(v) for k, v in items()}
        except Exception:
            pass
    return str(value)


def safe_count_parameters(
        parameters: Iterable[torch.nn.Parameter]) -> Tuple[int, int]:
    """Count initialized parameters without materializing LazyModule weights."""
    total = 0
    trainable = 0
    for parameter in parameters:
        if isinstance(parameter, UninitializedParameter):
            continue
        count = parameter.numel()
        total += count
        if parameter.requires_grad:
            trainable += count
    return int(total), int(trainable)


def _scalar_float(value, default=None):
    if value is None:
        return default
    if torch.is_tensor(value):
        if value.numel() == 0:
            return default
        value = value.detach().cpu().reshape(-1)[0].item()
    try:
        return float(value)
    except Exception:
        return default


def build_semantic_wandb_metadata(
        trainer,
        datamodule,
        model_hparams,
        num_classes: int,
        class_names: List[str],
        stuff_classes=None,
        bsr_enabled: bool = False,
        parameter_counts: Tuple[int, int] = None,
        commit_hash: str = None) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Build dataset-agnostic run metadata for W&B config and summary."""
    train_dataset = getattr(datamodule, 'train_dataset', None)
    val_dataset = getattr(datamodule, 'val_dataset', None)
    test_dataset = getattr(datamodule, 'test_dataset', None)
    dataset_ref = train_dataset or val_dataset or test_dataset
    datamodule_hparams = getattr(datamodule, 'hparams', None)
    dataloader_hparams = _cfg_get(datamodule_hparams, 'dataloader', None)
    optimizer = _cfg_get(model_hparams, 'optimizer', None)
    scheduler = _cfg_get(model_hparams, 'scheduler', None)
    bsr_cfg = _cfg_get(model_hparams, 'bsr', {}) or {}
    bsr_selector_cfg = _cfg_get(bsr_cfg, 'selector', {}) or {}
    bsr_refiner_cfg = _cfg_get(bsr_cfg, 'refiner', {}) or {}
    bsr_propagation_cfg = _cfg_get(bsr_cfg, 'propagation', {}) or {}
    total_params, trainable_params = parameter_counts or (None, None)

    if stuff_classes is None and dataset_ref is not None:
        stuff_classes = getattr(dataset_ref, 'stuff_classes', None)

    config = {
        'run/dataset_class': dataset_ref.__class__.__name__ if dataset_ref is not None else None,
        'run/num_classes': int(num_classes) if num_classes is not None else None,
        'run/class_names': list(class_names or []),
        'run/stuff_classes': list(stuff_classes or []),
        'run/train_items': _safe_len(train_dataset),
        'run/val_items': _safe_len(val_dataset),
        'run/test_items': _safe_len(test_dataset),
        'run/max_epochs': getattr(trainer, 'max_epochs', None),
        'run/check_val_every_n_epoch': getattr(trainer, 'check_val_every_n_epoch', None),
        'run/precision': getattr(trainer, 'precision', None),
        'run/batch_size': _cfg_get(dataloader_hparams, 'batch_size', None),
        'run/num_workers': _cfg_get(dataloader_hparams, 'num_workers', None),
        'run/xy_tiling': _cfg_get(datamodule_hparams, 'xy_tiling', None),
        'run/pc_tiling': _cfg_get(datamodule_hparams, 'pc_tiling', None),
        'run/voxel': _cfg_get(datamodule_hparams, 'voxel', None),
        'run/optimizer': _callable_name(optimizer),
        'run/lr': _cfg_get_nested(optimizer, ['keywords', 'lr'], None),
        'run/weight_decay': _cfg_get_nested(optimizer, ['keywords', 'weight_decay'], None),
        'run/scheduler': _callable_name(scheduler),
        'run/scheduler_warmup': _cfg_get_nested(scheduler, ['keywords', 'num_warmup'], None),
        'run/bsr_enabled': bool(bsr_enabled),
        'run/bsr_variant': _cfg_get(bsr_refiner_cfg, 'variant', None),
        'run/bsr_propagation_mode': _cfg_get(
            bsr_propagation_cfg, 'mode', 'slot_all_points' if bsr_enabled else None),
        'run/bsr_metric_level': _cfg_get(
            bsr_propagation_cfg, 'metric_level', 'point' if bsr_enabled else None),
        'run/bsr_topk_ratio': _cfg_get(bsr_selector_cfg, 'topk_ratio', None),
        'run/bsr_n_sample': _cfg_get(bsr_refiner_cfg, 'n_sample', None),
        'run/bsr_n_subregions': _cfg_get(bsr_refiner_cfg, 'n_subregions', None),
        'run/num_parameters': total_params,
        'run/num_trainable_parameters': trainable_params,
        'run/commit_hash': commit_hash,
    }
    config = {k: _wandb_safe_value(v) for k, v in config.items()}

    summary_keys = [
        'run/train_items',
        'run/val_items',
        'run/test_items',
        'run/max_epochs',
        'run/check_val_every_n_epoch',
        'run/num_parameters',
        'run/num_trainable_parameters',
    ]
    summary = {k: config[k] for k in summary_keys if config.get(k) is not None}
    summary['run/bsr_enabled'] = int(bool(bsr_enabled))
    return config, summary


def _point_attr_dim(value) -> int:
    if value is None or not torch.is_tensor(value):
        return 0
    if value.dim() <= 1:
        return 1
    return int(value.shape[-1])


def _resolve_available_bsr_raw_keys(
        data_0: Data,
        preferred_keys: List[str],
        expected_d_raw: int = None) -> List[str]:
    available = {}
    for key in preferred_keys:
        value = getattr(data_0, key, None)
        dim = _point_attr_dim(value)
        if dim > 0:
            available[key] = dim

    if 'pos' not in available:
        pos_value = getattr(data_0, 'pos', None)
        pos_dim = _point_attr_dim(pos_value)
        if pos_dim > 0:
            available['pos'] = pos_dim

    if not available:
        return []

    ordered_keys = []
    if 'pos' in available:
        ordered_keys.append('pos')
    ordered_keys.extend([key for key in preferred_keys if key in available and key != 'pos'])
    ordered_keys.extend([key for key in available.keys() if key not in ordered_keys])

    if expected_d_raw is None or expected_d_raw <= 0:
        return ordered_keys

    resolved = []
    resolved_dim = 0
    for key in ordered_keys:
        key_dim = available[key]
        if resolved_dim + key_dim > expected_d_raw:
            continue
        resolved.append(key)
        resolved_dim += key_dim
        if resolved_dim == expected_d_raw:
            break

    if resolved_dim == expected_d_raw and resolved:
        return resolved

    if 'pos' in available and available['pos'] == expected_d_raw:
        return ['pos']

    return ordered_keys


class SemanticSegmentationModule(LightningModule):
    """A LightningModule for semantic segmentation of point clouds.

    :param net: torch.nn.Module
        Backbone model. This can typically be an `SPT` object
    :param criterion: torch.nn._Loss
        Loss
    :param optimizer: torch.optim.Optimizer
        Optimizer
    :param scheduler: torch.optim.lr_scheduler.LRScheduler
        Learning rate scheduler
    :param num_classes: int
        Number of classes in the dataset
    :param class_names: List[str]
        Name for each class
    :param sampling_loss:  bool
        If True, the target labels will be obtained from labels of
        the points sampled in the batch at hand. This affects
        training supervision where sampling augmentations may be
        used for dropping some points or superpoints. If False, the
        target labels will be based on exact superpoint-wise
        histograms of labels computed at preprocessing time,
        disregarding potential level-0 point down-sampling.
        Not compatible with `net.nano=True` which avoids loading
        atom level.
    :param loss_type: str
        Type of loss applied.
        'ce': cross-entropy (if `multi_stage_loss_lambdas` is used,
        all 1+ levels will be supervised with cross-entropy).
        'kl': Kullback-Leibler divergence (if `multi_stage_loss_lambdas`
        is used, all 1+ levels will be supervised with cross-entropy).
        'ce_kl': cross-entropy on level 1 and Kullback-Leibler for
        all levels above
        'wce': not documented for now
        'wce_kl': not documented for now
    :param weighted_loss: bool
        If True, the loss will be weighted based on the class
        frequencies computed on the train dataset. See
        `BaseDataset.get_class_weight()` for more
    :param init_linear: str
        Initialization method for all linear layers. Supports
        'xavier_uniform', 'xavier_normal', 'kaiming_uniform',
        'kaiming_normal', 'trunc_normal'
    :param init_rpe: str
        Initialization method for all linear layers producing
        relative positional encodings. Supports 'xavier_uniform',
        'xavier_normal', 'kaiming_uniform', 'kaiming_normal',
        'trunc_normal'
    :param transformer_lr_scale: float
        Scaling parameter applied to the learning rate for the
        `TransformerBlock` in each `Stage` and for the pooling block
        in `DownNFuseStage` modules. Setting this to a value lower
        than 1 mitigates exploding gradients in attentive blocks
        during training
    :param multi_stage_loss_lambdas: List[float]
        List of weights for combining losses computed on the output
        of each partition level. If not specified, the loss will
        be computed on the level 1 outputs only
    :param gc_every_n_steps: int
        Explicitly call the garbage collector after a certain number
        of steps. May involve a computation overhead. Mostly hear
        for debugging purposes when observing suspicious GPU memory
        increase during training
    :param track_val_every_n_epoch: int
        If specified, the output for a validation batch of interest
        specified with `track_val_idx` will be stored to disk every
        `track_val_every_n_epoch` epochs. Must be a multiple of
        `check_val_every_n_epoch`. See `track_batch()` for more
    :param track_val_idx: int
        If specified, the output for the `track_val_idx`th
        validation batch will be saved to disk periodically based on
        `track_val_every_n_epoch`. If `track_test_idx=-1`, predictions
        for the entire test set will be saved to disk.
        Importantly, this index is expected to match the `Dataloader`'s
        index wrt the current epoch and NOT an index wrt the `Dataset`.
        Said otherwise, if the `Dataloader(shuffle=True)` then, the
        stored batch will not be the same at each epoch. For this
        reason, if tracking the same object across training is needed,
        the `Dataloader` and the transforms should be free from any
        stochasticity
    :param track_test_idx:
        If specified, the output for the `track_test_idx`th
        test batch will be saved to disk. If `track_test_idx=-1`,
        predictions for the entire test set will be saved to disk
    :param kwargs: Dict
        Kwargs will be passed to `_load_from_checkpoint()`
    """

    _IGNORED_HYPERPARAMETERS = [
        'net',
        'criterion',
        'partition',
        'partition_criterion']

    def __init__(
            self,
            net: torch.nn.Module,
            criterion: 'torch.nn._Loss',
            optimizer: torch.optim.Optimizer,
            scheduler: Any,
            num_classes: int,
            class_names: List[str] = None,
            sampling_loss: bool = False,
            loss_type: str = 'ce_kl',
            weighted_loss: bool = True,
            init_linear: str = None,
            init_rpe: str = None,
            transformer_lr_scale: float = 1,
            multi_stage_loss_lambdas: List[float] = None,
            gc_every_n_steps: int = 0,
            track_val_every_n_epoch: int = 1,
            track_val_idx: int = None,
            track_test_idx: int = None,
            bsr: dict = None,
            **kwargs):
        legacy_hspt_cfg = kwargs.pop('hspt', None)
        super().__init__()

        # Allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(
            logger=False, ignore=self._IGNORED_HYPERPARAMETERS)
    

        # Store the number of classes and the class names
        self.num_classes = num_classes
        self.class_names = class_names if class_names is not None \
            else [f'class-{i}' for i in range(num_classes)]

        # Loss function. If `multi_stage_loss_lambdas`, a MultiLoss is
        # built based on the input criterion
        if isinstance(criterion, MultiLoss):
            self.criterion = criterion
        elif multi_stage_loss_lambdas is not None:
            criteria = [
                deepcopy(criterion)
                for _ in range(len(multi_stage_loss_lambdas))]
            self.criterion = MultiLoss(criteria, multi_stage_loss_lambdas)
        else:
            self.criterion = criterion

        # Ignore the `num_classes` labels, which, by construction, are
        # where we send all 'ignored'/'void' annotations
        if isinstance(self.criterion, MultiLoss):
            for i in range(len(self.criterion.criteria)):
                self.criterion.criteria[i].ignore_index = num_classes
        else:
            self.criterion.ignore_index = num_classes

        # Network that will do the actual computation. NB, we make sure
        # the net returns the output from all up stages, if a multi-stage
        # loss is expected
        self.net = net
        if self.multi_stage_loss:
            self.net.output_stage_wise = True
            assert len(self.net.out_dim) == len(self.criterion), \
                f"The number of items in the multi-stage loss must match the " \
                f"number of stages in the net. Found " \
                f"{len(self.net.out_dim)} stages, but {len(self.criterion)} " \
                f"criteria in the loss."

        # Initialize the model segmentation head (or heads)
        if self.multi_stage_loss:
            self.head = ModuleList([
                Classifier(dim, num_classes) for dim in self.net.out_dim])
        else:
            self.head = Classifier(self.net.out_dim, num_classes)

        # Custom weight initialization. In particular, this applies
        # Xavier / Glorot initialization on Linear and RPE layers by
        # default, but can be tuned
        init = lambda m: init_weights(m, linear=init_linear, rpe=init_rpe)
        self.net.apply(init)
        self.head.apply(init)

        # If applicable, initialization of the CNN
        self.cnn_weights_initialization()
        
        # # Update version in the network for easier access during training 
        # # and inference.
        # self.net.version = src.__version__

        # Metric objects for calculating scores on each dataset split.
        # We add `ignore_index=num_classes` to account for
        # void/unclassified/ignored points, which are given
        # `num_classes` labels
        self.train_cm = ConfusionMatrix(num_classes)
        self.val_cm = ConfusionMatrix(num_classes)
        self.test_cm = ConfusionMatrix(num_classes)

        # For averaging loss across batches
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

        # For tracking best-so-far validation metrics
        self.val_miou_best = MaxMetric()
        self.val_oa_best = MaxMetric()
        self.val_macc_best = MaxMetric()
        self._val_best_epochs = {}

        # For tracking number of points andsuperpoints     
        self.val_n_p = SumMetric()
        self.test_n_p = SumMetric()
        
        self.val_n_sp = SumMetric()
        self.test_n_sp = SumMetric()
        
        # For tracking superpoint size histogram
        # It is suboptimal to use a CatMetric here, we could
        # use a Histogram instead, but torchmetrics does not support
        # histograms yet.

        # For tracking whether the test set has target labels. By
        # default, we assume the test set to have labels. But if a
        # single test batch misses labels, this will be set to False and
        # all test metrics computation will be skipped
        self.test_has_target = True

        # Explicitly call the garbage collector after a certain number
        # of steps
        self.gc_every_n_steps = int(gc_every_n_steps)
        self._wandb_run_metadata_logged = False

        if (bsr or {}).get('enable', False):
            self._geo_feature_keys = [
                'linearity', 'planarity', 'scattering', 'verticality',
                'curvature', 'length', 'surface', 'volume'
            ]
            self._geo_scatter_idx = 2
            self._geo_missing_warned = False

        # ===========================
        # BSR-SPT 模块初始化
        # ===========================
        self.bsr_enabled = False
        self.bsr = None
        self.bsr_config = bsr or {}
        self._bsr_raw_keys_resolved = False

        if self.bsr_config.get('enable', False):
            if not BSR_AVAILABLE:
                log.warning("BSR 模块未找到，跳过初始化")
            elif getattr(self.net, 'nano', False):
                log.warning("nano 模式不支持 BSR（需要原子点数据），跳过初始化")
            else:
                selector_cfg = self.bsr_config.get('selector', {})
                refiner_cfg = self.bsr_config.get('refiner', {})
                loss_cfg = self.bsr_config.get('loss', {})
                feedback_cfg = self.bsr_config.get('feedback', {})
                partition_cfg = self.bsr_config.get('partition_adapter', {})
                propagation_cfg = self.bsr_config.get('propagation', {})
                refiner_variant = refiner_cfg.get('variant', None)
                if refiner_variant is None:
                    legacy_mode = refiner_cfg.get('point_head_mode', 'residual_gated')
                    refiner_variant = {
                        'residual_gated': 'single_residual_gated',
                        'direct': 'single_direct',
                    }.get(legacy_mode, legacy_mode)
                requested_boundary_head = bool(refiner_cfg.get('use_boundary_head', True))
                lambda_boundary = float(loss_cfg.get('lambda_boundary', 0.0))
                effective_boundary_head = requested_boundary_head and lambda_boundary > 0.0
                if requested_boundary_head and not effective_boundary_head:
                    log.warning(
                        "BSR boundary head is disabled because lambda_boundary <= 0. "
                        "Set model.bsr.loss.lambda_boundary > 0 to train it."
                    )

                self.bsr_enabled = True
                self.bsr = BSRModule(
                    d_model=self.net.out_dim if not self.multi_stage_loss else self.net.out_dim[0],
                    num_classes=num_classes,
                    selector_topk_ratio=selector_cfg.get('topk_ratio', 0.2),
                    selector_score_terms=selector_cfg.get(
                        'score_terms', ['uncertainty', 'geometry', 'boundary']),
                    selector_term_weights=selector_cfg.get(
                        'term_weights',
                        {'uncertainty': 0.5, 'geometry': 0.25, 'boundary': 0.25}),
                    selector_scatter_idx=selector_cfg.get('scatter_idx', 2),
                    n_sample=refiner_cfg.get('n_sample', 64),
                    d_raw=refiner_cfg.get('d_raw', 6),
                    n_heads=refiner_cfg.get('n_heads', 4),
                    token_mode=refiner_cfg.get('token_mode', 'superpoint_query'),
                    dropout=refiner_cfg.get('dropout', 0.1),
                    hidden_dim=refiner_cfg.get('hidden_dim', None),
                    variant=refiner_variant,
                    n_subregions=refiner_cfg.get('n_subregions', 2),
                    assignment_temperature=refiner_cfg.get('assignment_temperature', 1.0),
                    point_head_mode=refiner_cfg.get('point_head_mode', None),
                    use_boundary_head=effective_boundary_head,
                    partition_adapter_enable=partition_cfg.get('enable', False),
                    partition_adapter_min_points=partition_cfg.get('min_points_per_sp', 3),
                    partition_adapter_merge_small=partition_cfg.get('merge_small_clusters', True),
                )
                self.bsr_raw_keys = self.bsr_config.get('raw_keys', ['pos', 'rgb'])
                self.bsr_sampling_mode = refiner_cfg.get('sampling_mode', 'coverage')
                self.bsr_sampling_without_replacement = refiner_cfg.get(
                    'sampling_without_replacement', True)
                self.bsr_lambda_refine = loss_cfg.get('lambda_refine', 0.5)
                self.bsr_lambda_consistency = loss_cfg.get('lambda_consistency', 0.1)
                self.bsr_lambda_diversity = float(loss_cfg.get('lambda_diversity', 0.02))
                self.bsr_lambda_boundary = lambda_boundary
                self.bsr_score_weighting = loss_cfg.get('score_weighting', True)
                self.bsr_score_weight_gamma = loss_cfg.get('score_weight_gamma', 1.0)
                self.bsr_loss_weight = float(loss_cfg.get('global_weight', 0.5))
                self.bsr_loss_warmup_epochs = int(loss_cfg.get('warmup_epochs', 10))
                self.bsr_feedback_warmup_epochs = int(
                    feedback_cfg.get('warmup_epochs', self.bsr_loss_warmup_epochs))
                self.bsr_logit_fusion_alpha = feedback_cfg.get('logit_fusion_alpha', 0.5)
                self.bsr_propagation_mode = propagation_cfg.get('mode', 'slot_all_points')
                self.bsr_propagation_chunk_size = int(propagation_cfg.get('chunk_size', 200000))
                self.bsr_store_slot_affinity = bool(propagation_cfg.get('store_slot_affinity', False))
                self.bsr_dual_slot_threshold = float(propagation_cfg.get('dual_slot_threshold', 0.2))
                self.bsr_metric_level = propagation_cfg.get('metric_level', 'point')
                self.bsr_selector_metric_terms = tuple(selector_cfg.get(
                    'score_terms', ['uncertainty', 'geometry', 'boundary']))

                self.train_refine_loss = MeanMetric()
                self.val_refine_loss = MeanMetric()
                self.test_refine_loss = MeanMetric()
                self.train_consistency_loss = MeanMetric()
                self.val_consistency_loss = MeanMetric()
                self.test_consistency_loss = MeanMetric()
                self.train_diversity_loss = MeanMetric()
                self.val_diversity_loss = MeanMetric()
                self.test_diversity_loss = MeanMetric()
                self.train_boundary_loss = MeanMetric()
                self.val_boundary_loss = MeanMetric()
                self.test_boundary_loss = MeanMetric()
                self.train_candidate_ratio = MeanMetric()
                self.val_candidate_ratio = MeanMetric()
                self.test_candidate_ratio = MeanMetric()
                self.train_candidate_score = MeanMetric()
                self.val_candidate_score = MeanMetric()
                self.test_candidate_score = MeanMetric()
                self.train_sample_valid_ratio = MeanMetric()
                self.val_sample_valid_ratio = MeanMetric()
                self.test_sample_valid_ratio = MeanMetric()
                self.train_num_valid_sampled_points = MeanMetric()
                self.val_num_valid_sampled_points = MeanMetric()
                self.test_num_valid_sampled_points = MeanMetric()
                self.train_avg_valid_points_per_candidate = MeanMetric()
                self.val_avg_valid_points_per_candidate = MeanMetric()
                self.test_avg_valid_points_per_candidate = MeanMetric()
                self.train_effective_refine_ratio = MeanMetric()
                self.val_effective_refine_ratio = MeanMetric()
                self.test_effective_refine_ratio = MeanMetric()
                self.train_point_gate_mean = MeanMetric()
                self.val_point_gate_mean = MeanMetric()
                self.test_point_gate_mean = MeanMetric()
                self.train_assignment_entropy = MeanMetric()
                self.val_assignment_entropy = MeanMetric()
                self.test_assignment_entropy = MeanMetric()
                self.train_secondary_slot_mass = MeanMetric()
                self.val_secondary_slot_mass = MeanMetric()
                self.test_secondary_slot_mass = MeanMetric()
                self.train_dual_slot_activation_ratio = MeanMetric()
                self.val_dual_slot_activation_ratio = MeanMetric()
                self.test_dual_slot_activation_ratio = MeanMetric()
                self.train_slot_diversity = MeanMetric()
                self.val_slot_diversity = MeanMetric()
                self.test_slot_diversity = MeanMetric()
                self.train_point_propagation_coverage = MeanMetric()
                self.val_point_propagation_coverage = MeanMetric()
                self.test_point_propagation_coverage = MeanMetric()
                self.train_bsr_selector_terms = ModuleDict({
                    term: MeanMetric() for term in self.bsr_selector_metric_terms})
                self.val_bsr_selector_terms = ModuleDict({
                    term: MeanMetric() for term in self.bsr_selector_metric_terms})
                self.test_bsr_selector_terms = ModuleDict({
                    term: MeanMetric() for term in self.bsr_selector_metric_terms})
                self.train_bsr_selector_term_vars = ModuleDict({
                    term: MeanMetric() for term in self.bsr_selector_metric_terms})
                self.val_bsr_selector_term_vars = ModuleDict({
                    term: MeanMetric() for term in self.bsr_selector_metric_terms})
                self.test_bsr_selector_term_vars = ModuleDict({
                    term: MeanMetric() for term in self.bsr_selector_metric_terms})

                self._bsr_fail_count = 0
                self._bsr_total_count = 0
                self._bsr_consecutive_fail = 0
                self._bsr_fuse_threshold = 10
                self._bsr_stage_fail_count = {'train': 0, 'val': 0, 'test': 0}
                self._bsr_stage_total_count = {'train': 0, 'val': 0, 'test': 0}

                log.info(
                    "BSR-SPT enabled: topk_ratio=%s, n_sample=%s, sampling_mode=%s, refiner_variant=%s, "
                    "lambda_refine=%s, lambda_consistency=%s, lambda_diversity=%s, lambda_boundary=%s, "
                    "score_weighting=%s, global_weight=%s, warmup_epochs=%s, fusion_warmup_epochs=%s",
                    selector_cfg.get('topk_ratio', 0.2),
                    refiner_cfg.get('n_sample', 64),
                    self.bsr_sampling_mode,
                    refiner_variant,
                    self.bsr_lambda_refine,
                    self.bsr_lambda_consistency,
                    self.bsr_lambda_diversity,
                    self.bsr_lambda_boundary,
                    self.bsr_score_weighting,
                    self.bsr_loss_weight,
                    self.bsr_loss_warmup_epochs,
                    self.bsr_feedback_warmup_epochs,
                )

        if legacy_hspt_cfg is not None:
            requested = bool((legacy_hspt_cfg or {}).get('enable', False))
            log.warning(
                "Legacy model.hspt configuration was provided%s, but the current "
                "semantic training/inference path ignores H-SPT and only supports "
                "SPT baseline or BSR-SPT.",
                " with enable=True" if requested else "",
            )

    def _current_bsr_stage(self) -> str:
        try:
            if self.trainer is None:
                return 'train' if self.training else 'val'
            if self.trainer.training:
                return 'train'
            if self.trainer.validating:
                return 'val'
            if self.trainer.testing:
                return 'test'
        except Exception:
            pass
        return 'train' if self.training else 'val'

    def _bsr_warmup_factor(self, warmup_epochs: int) -> float:
        if warmup_epochs <= 0:
            return 1.0
        epoch = max(int(getattr(self, 'current_epoch', 0)), 0)
        return min(1.0, float(epoch) / float(warmup_epochs))

    def _current_bsr_loss_weight(self) -> float:
        base_weight = float(getattr(self, 'bsr_loss_weight', 1.0))
        return base_weight * self._bsr_warmup_factor(
            int(getattr(self, 'bsr_loss_warmup_epochs', 0)))

    def _current_bsr_fusion_alpha(self) -> float:
        base_alpha = float(getattr(self, 'bsr_logit_fusion_alpha', 0.0))
        return base_alpha * self._bsr_warmup_factor(
            int(getattr(self, 'bsr_feedback_warmup_epochs', 0)))

    def _current_bsr_point_gate_mean(self, bsr_output, device: torch.device) -> torch.Tensor:
        gates = getattr(bsr_output, 'point_residual_gates', None)
        if gates is None or gates.numel() == 0:
            return torch.tensor(0.0, device=device)

        gates = gates.to(device=device, dtype=torch.float32)
        sampled_mask = getattr(bsr_output, 'sampled_point_mask', None)
        if sampled_mask is None or sampled_mask.numel() == 0:
            return gates.mean()

        valid = sampled_mask.to(device=device, dtype=torch.float32)
        denom = valid.sum().clamp(min=1.0)
        return (gates * valid).sum() / denom

    def _current_bsr_output_mean(
            self,
            bsr_output,
            attr: str,
            device: torch.device) -> torch.Tensor:
        values = getattr(bsr_output, attr, None)
        if values is None:
            return torch.tensor(0.0, device=device)
        if isinstance(values, (float, int)):
            return torch.tensor(float(values), device=device)
        if not torch.is_tensor(values) or values.numel() == 0:
            return torch.tensor(0.0, device=device)
        return values.detach().to(device=device, dtype=torch.float32).mean()

    def _fuse_bsr_superpoint_logits(
            self,
            coarse_logits: torch.Tensor,
            bsr_output) -> torch.Tensor:
        """Fuse refined candidate logits back into superpoint logits."""
        alpha = self._current_bsr_fusion_alpha()
        if alpha <= 0.0 or bsr_output is None:
            return coarse_logits

        refined_logits = getattr(bsr_output, 'refined_sp_logits', None)
        candidate_indices = getattr(bsr_output, 'candidate_indices', None)
        if refined_logits is None or candidate_indices is None or candidate_indices.numel() == 0:
            return coarse_logits

        fused_logits = coarse_logits.clone()
        fused_logits[candidate_indices] = (
            (1.0 - alpha) * fused_logits[candidate_indices]
            + alpha * refined_logits[candidate_indices].to(fused_logits.dtype)
        )
        return fused_logits

    def _scatter_superpoint_logits_to_points(
            self,
            logits: torch.Tensor,
            nag: NAG) -> torch.Tensor:
        super_index = getattr(nag[0], 'super_index', None)
        if super_index is None:
            raise ValueError("NAG level-0 data must provide super_index for point-level metrics")
        return logits[super_index.to(device=logits.device, dtype=torch.long)]

    def _build_bsr_point_logits(
            self,
            nag: NAG,
            superpoint_logits: torch.Tensor,
            bsr_output) -> None:
        """Construct paper-aligned level-0 logits for BSR output.

        Candidate points receive slot-propagated logits; non-candidate
        points inherit their parent superpoint logits.
        """
        super_index = getattr(nag[0], 'super_index', None)
        if super_index is None:
            return

        super_index = super_index.to(device=superpoint_logits.device, dtype=torch.long)
        point_logits = superpoint_logits[super_index].detach().clone()
        propagation_mask = torch.zeros(
            super_index.numel(),
            dtype=torch.bool,
            device=superpoint_logits.device,
        )
        bsr_output.point_logits = point_logits
        bsr_output.point_propagation_mask = propagation_mask
        bsr_output.point_propagation_coverage = 0.0

        candidate_indices = getattr(bsr_output, 'candidate_indices', None)
        if candidate_indices is None or candidate_indices.numel() == 0:
            return

        mode = str(getattr(self, 'bsr_propagation_mode', 'slot_all_points'))
        alpha = float(getattr(bsr_output, 'warmup_fusion_alpha', self._current_bsr_fusion_alpha()))
        coarse_candidate_logits = getattr(bsr_output, 'coarse_candidate_logits', None)
        if coarse_candidate_logits is None:
            coarse_candidate_logits = superpoint_logits[candidate_indices]

        if mode in {'none', 'legacy_superpoint'}:
            bsr_output.point_propagation_coverage = 0.0
            return

        if mode == 'sampled_point_only':
            sampled_idx = getattr(bsr_output, 'sampled_point_indices', None)
            sampled_mask = getattr(bsr_output, 'sampled_point_mask', None)
            sampled_logits = getattr(bsr_output, 'sampled_point_logits', None)
            if sampled_idx is None or sampled_logits is None or sampled_idx.numel() == 0:
                return
            valid = sampled_idx >= 0
            if sampled_mask is not None:
                valid = valid & sampled_mask.to(device=valid.device, dtype=torch.bool)
            if not valid.any():
                return

            row_ids = torch.arange(sampled_idx.shape[0], device=sampled_idx.device).unsqueeze(1)
            row_ids = row_ids.expand_as(sampled_idx)[valid]
            point_ids = sampled_idx[valid].to(device=point_logits.device, dtype=torch.long)
            valid_point_ids = (point_ids >= 0) & (point_ids < point_logits.shape[0])
            if not valid_point_ids.any():
                return
            row_ids = row_ids[valid_point_ids].to(device=point_logits.device, dtype=torch.long)
            point_ids = point_ids[valid_point_ids]
            local_logits = sampled_logits[valid][valid_point_ids].to(device=point_logits.device, dtype=point_logits.dtype)
            coarse_base = coarse_candidate_logits[row_ids].to(device=point_logits.device, dtype=point_logits.dtype)
            point_logits[point_ids] = (1.0 - alpha) * coarse_base + alpha * local_logits
            propagation_mask[point_ids] = True
            bsr_output.point_logits = point_logits
            bsr_output.point_propagation_mask = propagation_mask
            bsr_output.point_propagation_coverage = float(
                propagation_mask.float().mean().detach().item()
            )
            return

        slot_tokens = getattr(bsr_output, 'slot_tokens', None)
        slot_logits = getattr(bsr_output, 'slot_logits', None)
        if slot_tokens is None or slot_logits is None or slot_tokens.numel() == 0 or slot_logits.numel() == 0:
            return

        point_indices, candidate_rows, raw_points = build_bsr_candidate_point_cloud(
            nag=nag,
            raw_keys=getattr(self, 'bsr_raw_keys', ['pos', 'rgb']),
            candidate_indices=candidate_indices,
            device=superpoint_logits.device,
        )
        if point_indices.numel() == 0:
            return

        chunk_size = max(int(getattr(self, 'bsr_propagation_chunk_size', 200000)), 1)
        store_affinity = bool(getattr(self, 'bsr_store_slot_affinity', False))
        slot_affinity = None
        if store_affinity:
            slot_affinity = point_logits.new_zeros((point_logits.shape[0], slot_logits.shape[1]))

        with torch.no_grad():
            for start in range(0, point_indices.numel(), chunk_size):
                end = min(start + chunk_size, point_indices.numel())
                rows = candidate_rows[start:end].to(device=superpoint_logits.device, dtype=torch.long)
                points = point_indices[start:end].to(device=superpoint_logits.device, dtype=torch.long)
                raw_chunk = raw_points[start:end].to(device=superpoint_logits.device, dtype=superpoint_logits.dtype)
                centroids = nag[1].pos.to(device=superpoint_logits.device, dtype=superpoint_logits.dtype)[
                    candidate_indices.to(device=superpoint_logits.device, dtype=torch.long)[rows]
                ]
                point_tokens = self.bsr.refiner.encode_candidate_points(raw_chunk, centroids)
                local_logits, affinity = self.bsr.refiner.propagate_slot_logits(
                    point_tokens=point_tokens,
                    slot_tokens=slot_tokens.to(device=superpoint_logits.device, dtype=point_tokens.dtype)[rows],
                    slot_logits=slot_logits.to(device=superpoint_logits.device, dtype=point_tokens.dtype)[rows],
                )
                coarse_base = coarse_candidate_logits.to(
                    device=superpoint_logits.device,
                    dtype=local_logits.dtype,
                )[rows]
                point_logits[points] = (
                    (1.0 - alpha) * coarse_base + alpha * local_logits
                ).to(dtype=point_logits.dtype)
                propagation_mask[points] = True
                if slot_affinity is not None:
                    slot_affinity[points] = affinity.to(dtype=slot_affinity.dtype)

        bsr_output.point_logits = point_logits
        bsr_output.point_propagation_mask = propagation_mask
        bsr_output.point_propagation_coverage = float(
            propagation_mask.float().mean().detach().item()
        )
        bsr_output.slot_affinity = slot_affinity

    def _update_semantic_confusion_matrix(
            self,
            cm: ConfusionMatrix,
            output: SemanticSegmentationOutput) -> None:
        if (
            getattr(self, 'bsr_metric_level', 'point') == 'point'
            and getattr(output, 'point_y', None) is not None
            and (
                getattr(output, 'point_logits', None) is not None
                or getattr(output, 'super_index', None) is not None
            )
        ):
            cm(
                output.point_semantic_pred().detach(),
                output.point_y.detach())
            return

        cm(
            output.semantic_pred().detach(),
            output.semantic_target.detach())

    def _resolve_bsr_raw_keys(self, nag: NAG) -> None:
        if self._bsr_raw_keys_resolved:
            return

        preferred_keys = list(getattr(self, 'bsr_raw_keys', ['pos', 'rgb']))
        expected_d_raw = None
        if self.bsr is not None and getattr(self.bsr, 'refiner', None) is not None:
            encoder_layer = self.bsr.refiner.point_encoder[0]
            encoder_weight = getattr(encoder_layer, 'weight', None)
            if torch.is_tensor(encoder_weight) and encoder_weight.ndim == 2:
                expected_d_raw = int(encoder_weight.shape[1])
            else:
                in_features = getattr(encoder_layer, 'in_features', None)
                if isinstance(in_features, int) and in_features > 0:
                    expected_d_raw = int(in_features)

        resolved_keys = _resolve_available_bsr_raw_keys(
            nag[0],
            preferred_keys=preferred_keys,
            expected_d_raw=expected_d_raw,
        )

        if not resolved_keys:
            raise ValueError("BSR could not resolve any raw_keys from level-0 attributes")

        if expected_d_raw is not None:
            resolved_d_raw = sum(_point_attr_dim(getattr(nag[0], key, None)) for key in resolved_keys)
            if resolved_keys != preferred_keys:
                log.warning(
                    "BSR: adapted raw point attributes for checkpoint-compatible d_raw=%s: %s -> %s",
                    expected_d_raw,
                    preferred_keys,
                    resolved_keys,
                )
            if resolved_d_raw != expected_d_raw:
                log.warning(
                    "BSR: resolved raw point attributes %s produce d_raw=%s, "
                    "but the refiner expects d_raw=%s. Forward may fall back to baseline.",
                    resolved_keys,
                    resolved_d_raw,
                    expected_d_raw,
                )

        self.bsr_raw_keys = resolved_keys
        self._bsr_raw_keys_resolved = True
        log.info("BSR: resolved raw point attributes for packing: %s", self.bsr_raw_keys)

    def _extract_handcrafted_features(self, nag) -> 'Optional[torch.Tensor]':
        """
        从 NAG Level 1 提取几何手工特征，返回固定顺序的张量。
        
        通道定义契约：[linearity, planarity, scattering, verticality, curvature, length, surface, volume]
        缺失的字段补 0，且只在首次缺失时记录一次 warning。
        
        参数:
            nag: NAG 对象
        
        返回:
            handcrafted_features: [M, D_geo] 张量，或 None（无任何可用特征时）
        """
        if not hasattr(self, '_geo_feature_keys'):
            return None
        
        sp_data = nag[1]
        device = sp_data.pos.device
        result, missing_keys = extract_selector_handcrafted_features(
            nag=nag,
            preferred_keys=self._geo_feature_keys,
            device=device,
        )
        
        # 首次缺失时记录 warning
        if missing_keys and not self._geo_missing_warned:
            log.warning(
                f"候选先验所需几何特征缺失 {missing_keys}，对应通道补 0。"
                f"如果全部缺失，几何项将被跳过，仅保留语义不确定性与边界代理。"
            )
            self._geo_missing_warned = True

        # 如果全部缺失，则返回 None，避免几何项对候选先验产生伪信号
        if len(missing_keys) == len(self._geo_feature_keys):
            return None
        
        return result

    def cnn_weights_initialization(self) -> None:
        if not getattr(self.net.first_stage, 'cnn_blocks', False):
            log.info("The first stage does not use a CNN.")
            return

        # Random initialization of the CNN
        if getattr(self.hparams, 'train_cnn_from_scratch', False):
            assert not (getattr(self.hparams, 'freeze_cnn', False)), \
                "Are you sure you want to freeze a randomly initialized CNN ?"
            log.info("The CNN is trained from scratch")
            return

        # Initialization of the CNN from a pretrained checkpoint
        ckpt_path = getattr(self.hparams, 'pretrained_cnn_ckpt_path', None)
        assert ckpt_path is not None and os.path.exists(ckpt_path), \
            ("A pretrained CNN checkpoint must be provided if "
             "you don't want to train the CNN from scratch. "
             "Currently, `train_cnn_from_scratch=False` and "
             f"`pretrained_cnn_ckpt_path={ckpt_path}`")

        log.info(
            f"Initializing CNN of semantic module with the pretrained "
            f"checkpoint (produced by the partition training): {ckpt_path}")

        self.net.first_stage = PretrainedCNN.load_checkpoint(
            self.net.first_stage,
            ckpt_path,
            self.device,
            verbose=False)

        if getattr(self.hparams, 'freeze_cnn', False):
            log.info("The pretrained CNN will be frozen during training.")
            self.net.first_stage.cnn_blocks.freeze()
        else :
            log.info("The pretrained CNN will NOT be frozen during training.")

    def forward(self, nag: NAG) -> SemanticSegmentationOutput:
        """
        前向传播。
        
        若启用 BSR，会执行以下额外操作：
        1. 使用 Boundary Prior Selector 选择高风险超点
        2. 使用真正的 superpoint-query / point-key-value cross-attention 细化候选区域
        3. 将局部细化结果反馈到 superpoint 语义预测
        """
        if self.bsr_enabled:
            self._resolve_bsr_raw_keys(nag)

        x = self.net(nag)
        
        # 获取超点特征（用于主分类头和局部细化模块）
        sp_features = x[0] if self.multi_stage_loss else x
        
        # 计算粗分类 logits
        coarse_logits = self.head[0](sp_features) if self.multi_stage_loss else self.head(sp_features)
        
        # ===========================
        # BSR 集成
        # ===========================
        bsr_output = None
        if self.bsr_enabled and self.bsr is not None:
            bsr_stage = self._current_bsr_stage()
            self._bsr_total_count += 1
            if bsr_stage in self._bsr_stage_total_count:
                self._bsr_stage_total_count[bsr_stage] += 1
            try:
                sp_centroids = nag[1].pos if hasattr(nag[1], 'pos') else None
                if sp_centroids is not None:
                    handcrafted_feats = self._extract_handcrafted_features(nag)
                    edge_index = nag[1].edge_index if hasattr(nag[1], 'edge_index') else None
                    candidate_indices, candidate_scores, selector_score_terms = self.bsr.select_candidates(
                        coarse_logits=coarse_logits,
                        handcrafted_features=handcrafted_feats,
                        edge_index=edge_index,
                    )
                    candidate_raw_points, candidate_point_idx, candidate_mask = build_bsr_packed_points(
                        nag,
                        n_sample=self.bsr.n_sample,
                        raw_keys=self.bsr_raw_keys,
                        superpoint_indices=candidate_indices,
                        device=sp_features.device,
                        sampling_mode=self.bsr_sampling_mode,
                        sampling_without_replacement=self.bsr_sampling_without_replacement,
                    )
                    bsr_output = self.bsr.refine_candidates(
                        candidate_indices=candidate_indices,
                        candidate_scores=candidate_scores,
                        sp_features=sp_features,
                        sp_centroids=sp_centroids,
                        coarse_logits=coarse_logits,
                        candidate_raw_points=candidate_raw_points,
                        candidate_point_indices=candidate_point_idx,
                        candidate_point_mask=candidate_mask,
                        selector_score_terms=selector_score_terms,
                        sampling_mode=self.bsr_sampling_mode,
                    )
                    bsr_output.warmup_loss_weight = self._current_bsr_loss_weight()
                    bsr_output.warmup_fusion_alpha = self._current_bsr_fusion_alpha()
                    if bsr_output.refined_sp_features is not None:
                        sp_features = bsr_output.refined_sp_features
                    if bsr_output.candidate_indices is not None and bsr_output.candidate_indices.numel() > 0:
                        bsr_output.coarse_candidate_logits = coarse_logits[
                            bsr_output.candidate_indices
                        ].clone()
                    coarse_logits = self._fuse_bsr_superpoint_logits(coarse_logits, bsr_output)
                    if bsr_output.candidate_indices is not None and bsr_output.candidate_indices.numel() > 0:
                        bsr_output.fused_candidate_logits = coarse_logits[
                            bsr_output.candidate_indices
                        ]
                    self._build_bsr_point_logits(nag, coarse_logits, bsr_output)
                    self._bsr_consecutive_fail = 0
            except Exception as e:
                self._bsr_fail_count += 1
                self._bsr_consecutive_fail += 1
                if bsr_stage in self._bsr_stage_fail_count:
                    self._bsr_stage_fail_count[bsr_stage] += 1
                if self.training and self._bsr_consecutive_fail >= self._bsr_fuse_threshold:
                    raise RuntimeError(
                        f"BSR consecutive failures reached {self._bsr_consecutive_fail}. "
                        f"Latest error: {e}"
                    )
                log.warning(
                    "BSR forward failed (%s/%s): %s. Falling back to baseline logits.",
                    self._bsr_fail_count,
                    self._bsr_total_count,
                    e,
                )
                bsr_output = None

        # 构建最终 logits
        if self.multi_stage_loss:
            logits = [coarse_logits] + [head(x_) for head, x_ in zip(self.head[1:], x[1:])]
        else:
            logits = coarse_logits
        
        output = SemanticSegmentationOutput(logits)
        
        if self.net.store_features:
            output.x = x
        
        # 附加局部细化输出（供 model_step 计算附加损失）
        if bsr_output is not None:
            output.bsr_output = bsr_output
            if getattr(bsr_output, 'point_logits', None) is not None:
                output.point_logits = bsr_output.point_logits
        
        return output

    def _attach_bsr_tracking_metadata(self, batch: NAG, bsr_output) -> None:
        """Attach dense BSR diagnostics to tracked NAG predictions."""
        candidate_indices = getattr(bsr_output, 'candidate_indices', None)
        if candidate_indices is None:
            return

        num_superpoints = batch[1].num_nodes
        device = batch[1].semantic_pred.device if hasattr(batch[1], 'semantic_pred') else candidate_indices.device
        candidate_indices = candidate_indices.to(device=device, dtype=torch.long)
        if candidate_indices.numel() == 0:
            return

        dense_candidate_mask = torch.zeros(num_superpoints, dtype=torch.bool, device=device)
        dense_candidate_mask[candidate_indices] = True

        def _scatter_candidate_values(values, dtype=torch.float32):
            dense = torch.zeros(num_superpoints, dtype=dtype, device=device)
            if values is None:
                return dense
            if torch.is_tensor(values):
                if values.numel() == 0:
                    return dense
                dense[candidate_indices] = values.to(device=device, dtype=dtype)
                return dense
            dense[candidate_indices] = torch.as_tensor(values, dtype=dtype, device=device)
            return dense

        dense_candidate_scores = _scatter_candidate_values(getattr(bsr_output, 'candidate_scores', None))
        dense_assignment_entropy = _scatter_candidate_values(getattr(bsr_output, 'assignment_entropy', None))
        dense_secondary_slot_mass = _scatter_candidate_values(getattr(bsr_output, 'secondary_slot_mass', None))
        dense_slot_diversity = _scatter_candidate_values(getattr(bsr_output, 'slot_diversity', None))
        dense_dual_slot_active = dense_secondary_slot_mass > 0.2

        batch[1].bsr_candidate_mask = dense_candidate_mask
        batch[1].bsr_candidate_score = dense_candidate_scores
        batch[1].bsr_assignment_entropy = dense_assignment_entropy
        batch[1].bsr_secondary_slot_mass = dense_secondary_slot_mass
        batch[1].bsr_dual_slot_active = dense_dual_slot_active
        batch[1].bsr_slot_diversity = dense_slot_diversity

        if getattr(batch[0], 'super_index', None) is not None:
            point_super_index = batch[0].super_index.to(device=device, dtype=torch.long)
            batch[0].bsr_candidate_mask = dense_candidate_mask[point_super_index]
            batch[0].bsr_candidate_score = dense_candidate_scores[point_super_index]
            batch[0].bsr_assignment_entropy = dense_assignment_entropy[point_super_index]
            batch[0].bsr_secondary_slot_mass = dense_secondary_slot_mass[point_super_index]
            batch[0].bsr_dual_slot_active = dense_dual_slot_active[point_super_index]
            batch[0].bsr_slot_diversity = dense_slot_diversity[point_super_index]

        point_logits = getattr(bsr_output, 'point_logits', None)
        if point_logits is not None:
            point_logits = point_logits.to(device=device)
            batch[0].bsr_point_logits = point_logits
            batch[0].bsr_point_semantic_pred = torch.argmax(point_logits, dim=1)

        point_mask = getattr(bsr_output, 'point_propagation_mask', None)
        if point_mask is not None:
            batch[0].bsr_point_propagation_mask = point_mask.to(device=device, dtype=torch.bool)

        coverage = float(getattr(bsr_output, 'point_propagation_coverage', 0.0))
        num_level0_points = getattr(batch[0], 'num_nodes', None)
        if num_level0_points is None:
            num_level0_points = batch[0].pos.shape[0]
        batch[0].bsr_point_propagation_coverage = torch.full(
            (int(num_level0_points),),
            coverage,
            dtype=torch.float32,
            device=device,
        )

        slot_affinity = getattr(bsr_output, 'slot_affinity', None)
        if slot_affinity is not None:
            batch[0].bsr_slot_affinity = slot_affinity.to(device=device, dtype=torch.float32)

    @property
    def multi_stage_loss(self) -> bool:
        return isinstance(self.criterion, MultiLoss)

    def _iter_wandb_loggers(self):
        trainer = getattr(self, 'trainer', None)
        loggers = getattr(trainer, 'loggers', None) if trainer is not None else None
        if loggers is None:
            loggers = [getattr(self, 'logger', None)]
        elif not isinstance(loggers, (list, tuple)):
            loggers = [loggers]
        for logger in loggers:
            if isinstance(logger, WandbLogger):
                yield logger

    def _count_parameters(self) -> Tuple[int, int]:
        return safe_count_parameters(self.parameters())

    def _log_wandb_run_metadata(self) -> None:
        if self._wandb_run_metadata_logged:
            return
        wandb_loggers = list(self._iter_wandb_loggers())
        if not wandb_loggers:
            return

        datamodule = getattr(self.trainer, 'datamodule', None)
        if datamodule is None:
            return

        dataset = (
            getattr(datamodule, 'train_dataset', None)
            or getattr(datamodule, 'val_dataset', None)
            or getattr(datamodule, 'test_dataset', None))
        stuff_classes = getattr(dataset, 'stuff_classes', None) if dataset is not None else None
        config, summary = build_semantic_wandb_metadata(
            trainer=self.trainer,
            datamodule=datamodule,
            model_hparams=self.hparams,
            num_classes=self.num_classes,
            class_names=self.class_names,
            stuff_classes=stuff_classes,
            bsr_enabled=getattr(self, 'bsr_enabled', False),
            parameter_counts=self._count_parameters(),
            commit_hash=get_commit_hash(),
        )

        metric_patterns = [
            'train/*',
            'val/*',
            'test/*',
            'lr-*',
            'run/*',
        ]
        for logger in wandb_loggers:
            experiment = logger.experiment
            try:
                experiment.define_metric('epoch')
                for pattern in metric_patterns:
                    experiment.define_metric(pattern, step_metric='epoch')
            except Exception as exc:
                log.warning("Could not define W&B metric axes: %s", exc)
            try:
                experiment.config.update(config, allow_val_change=True)
                experiment.summary.update(summary)
            except Exception as exc:
                log.warning("Could not update W&B run metadata: %s", exc)

        self._wandb_run_metadata_logged = True

    def _update_wandb_fit_summary(self) -> None:
        wandb_loggers = list(self._iter_wandb_loggers())
        if not wandb_loggers:
            return

        checkpoint_callback = getattr(self.trainer, 'checkpoint_callback', None)
        summary = {
            'run/completed_epochs': int(getattr(self, 'current_epoch', 0)),
            'run/global_step': int(getattr(self, 'global_step', 0)),
        }
        if checkpoint_callback is not None:
            summary['run/best_model_path'] = getattr(
                checkpoint_callback, 'best_model_path', '')
            summary['run/last_model_path'] = getattr(
                checkpoint_callback, 'last_model_path', '')
            best_model_score = getattr(checkpoint_callback, 'best_model_score', None)
            if best_model_score is not None:
                summary['run/best_model_score'] = _wandb_safe_value(best_model_score)

        for logger in wandb_loggers:
            try:
                logger.experiment.summary.update(summary)
            except Exception as exc:
                log.warning("Could not update W&B fit summary: %s", exc)

    def on_fit_start(self) -> None:
        # This is a bit of a late initialization for the LightningModule
        # At this point, we can access some LightningDataModule-related
        # parameters that were not available beforehand. So we take this
        # opportunity to catch the number of classes or class weights
        # from the LightningDataModule

        # Get the LightningDataModule number of classes and make sure it
        # matches self.num_classes. We could also forcefully update the
        # LightningModule with this new information, but it could easily
        # become tedious to track all places where num_classes affects
        # the LightningModule object.
        dataset = self.trainer.datamodule.test_dataset
        num_classes = dataset.num_classes
        assert num_classes == self.num_classes, \
            f'LightningModule has {self.num_classes} classes while the ' \
            f'LightningDataModule has {num_classes} classes.'

        self.class_names = dataset.class_names
        self._log_wandb_run_metadata()

        if not self.hparams.weighted_loss:
            return

        if not hasattr(self.criterion, 'weight'):
            log.warning(
                f"{self.criterion} does not have a 'weight' attribute. "
                f"Class weights will be ignored...")
            return

        # Set class weights for the criterion
        if not self.trainer.datamodule.hparams.prepare_only_test:
            weight = self.trainer.datamodule.train_dataset.get_class_weight(
                smooth=getattr(self.hparams, 'weighted_loss_smooth', 'sqrt'))
            self.criterion.weight = weight.to(self.device)

        # Check that the period of track_val_every_n_epoch` is a
        # multiple of check_val_every_n_epoch
        if self.trainer.check_val_every_n_epoch is not None:
            assert (self.hparams.track_val_every_n_epoch
                    % self.trainer.check_val_every_n_epoch == 0), \
                (f"Expected 'track_val_every_n_epoch' to be a multiple of "
                 f"'check_val_every_n_epoch', but received "
                 f"{self.hparams.track_val_every_n_epoch} and "
                 f"{self.trainer.check_val_every_n_epoch} instead.")

    def on_fit_end(self) -> None:
        self._update_wandb_fit_summary()

    def on_train_start(self) -> None:
        # By default, lightning executes validation step sanity checks
        # before training starts, so we need to make sure `*_best`
        # metrics do not store anything from these checks
        self.val_cm.reset()
        self.val_miou_best.reset()
        self.val_oa_best.reset()
        self.val_macc_best.reset()
        self._val_best_epochs.clear()

    def on_train_epoch_start(self) -> None:
        if getattr(self, 'bsr_enabled', False):
            self._bsr_consecutive_fail = 0
            self._bsr_stage_fail_count['train'] = 0
            self._bsr_stage_total_count['train'] = 0

    def on_validation_epoch_start(self) -> None:
        garbage_collection_cuda()
        if getattr(self, 'bsr_enabled', False):
            self._bsr_consecutive_fail = 0
            self._bsr_stage_fail_count['val'] = 0
            self._bsr_stage_total_count['val'] = 0

    def on_test_epoch_start(self) -> None:
        garbage_collection_cuda()
        if getattr(self, 'bsr_enabled', False):
            self._bsr_consecutive_fail = 0
            self._bsr_stage_fail_count['test'] = 0
            self._bsr_stage_total_count['test'] = 0

    def gc_collect(self) -> None:
        num_steps = self.trainer.fit_loop.epoch_loop._batches_that_stepped + 1
        period = self.gc_every_n_steps
        if period is None or period < 1:
            return
        if num_steps % period == 0:
            garbage_collection_cuda()

    def on_train_batch_start(self, *args) -> None:
        self.gc_collect()

    def on_validation_batch_start(self, *args) -> None:
        self.gc_collect()

    def on_test_batch_start(self, *args) -> None:
        self.gc_collect()

    def model_step(
            self,
            batch: NAG
    ) -> Tuple[torch.Tensor, SemanticSegmentationOutput]:
        # Forward step on the input batch. If a (NAG, Transform, int)
        # tuple is passed, the multi-run inference will be triggered
        output = self.step_single_run_inference(batch) \
            if isinstance(batch, NAG) \
            else self.step_multi_run_inference(*batch)

        # If the input batch does not have labels (e.g. test set with
        # held-out labels), y_hist will be None and the loss will not be
        # computed
        if not output.has_target:
            return None, output

        # Compute the loss either in a point-wise or segment-wise
        # fashion. Cross-Entropy with pointwise_loss is equivalent to
        # KL-divergence
        if self.multi_stage_loss:
            if self.hparams.loss_type == 'ce':
                loss = self.criterion(
                    output.logits, [y.argmax(dim=1) for y in output.y_hist])
            elif self.hparams.loss_type == 'wce':
                y_hist_dominant = []
                for y in output.y_hist:
                    y_dominant = y.argmax(dim=1)
                    y_hist_dominant_ = torch.zeros_like(y)
                    y_hist_dominant_[:, y_dominant] = y.sum(dim=1)
                    y_hist_dominant.append(y_hist_dominant_)
                loss = 0
                enum = zip(
                    self.criterion.lambdas,
                    self.criterion.criteria,
                    output.logits,
                    y_hist_dominant)
                for lamb, criterion, a, b in enum:
                    loss = loss + lamb * loss_with_target_histogram(
                        criterion, a, b)
            elif self.hparams.loss_type == 'ce_kl':
                loss = 0
                enum = zip(
                    self.criterion.lambdas,
                    self.criterion.criteria,
                    output.logits,
                    output.y_hist)
                for i, (lamb, criterion, a, b) in enumerate(enum):
                    if i == 0:
                        loss = loss + criterion(a, b.argmax(dim=1))
                        continue
                    loss = loss + lamb * loss_with_target_histogram(
                        criterion, a, b)
            elif self.hparams.loss_type == 'wce_kl':
                loss = 0
                enum = zip(
                    self.criterion.lambdas,
                    self.criterion.criteria,
                    output.logits,
                    output.y_hist)
                for i, (lamb, criterion, a, b) in enumerate(enum):
                    if i == 0:
                        y_dominant = b.argmax(dim=1)
                        y_hist_dominant = torch.zeros_like(b)
                        y_hist_dominant[:, y_dominant] = b.sum(dim=1)
                        loss = loss + loss_with_target_histogram(
                            criterion, a, y_hist_dominant)
                        continue
                    loss = loss + lamb * loss_with_target_histogram(
                        criterion, a, b)
            elif self.hparams.loss_type == 'kl':
                loss = 0
                enum = zip(
                    self.criterion.lambdas,
                    self.criterion.criteria,
                    output.logits,
                    output.y_hist)
                for lamb, criterion, a, b in enum:
                    loss = loss + lamb * loss_with_target_histogram(
                        criterion, a, b)
            else:
                raise ValueError(
                    f"Unknown multi-stage loss '{self.hparams.loss_type}'")
        else:
            if self.hparams.loss_type == 'ce':
                loss = self.criterion(output.logits, output.y_hist.argmax(dim=1))
            elif self.hparams.loss_type == 'wce':
                y_dominant = output.y_hist.argmax(dim=1)
                y_hist_dominant = torch.zeros_like(output.y_hist)
                y_hist_dominant[:, y_dominant] = output.y_hist.sum(dim=1)
                loss = loss_with_target_histogram(
                    self.criterion, output.logits, y_hist_dominant)
            elif self.hparams.loss_type == 'kl':
                loss = loss_with_target_histogram(
                    self.criterion, output.logits, output.y_hist)
            else:
                raise ValueError(
                    f"Unknown single-stage loss '{self.hparams.loss_type}'")

        # ===========================
        # BSR Selective Consistency Learning
        # ===========================
        if self.bsr_enabled and hasattr(output, 'bsr_output') and output.bsr_output is not None:
            try:
                if isinstance(batch, NAG) and hasattr(batch[0], 'y') and batch[0].y is not None:
                    gt_labels = batch[0].y
                    bsr_total_loss, bsr_losses = compute_bsr_losses(
                        bsr_output=output.bsr_output,
                        gt_labels=gt_labels,
                        num_classes=self.num_classes,
                        ignore_index=self.num_classes,
                        lambda_refine=self.bsr_lambda_refine,
                        lambda_consistency=self.bsr_lambda_consistency,
                        lambda_diversity=self.bsr_lambda_diversity,
                        lambda_boundary=self.bsr_lambda_boundary,
                        score_weighting=self.bsr_score_weighting,
                        score_weight_gamma=self.bsr_score_weight_gamma,
                    )
                    current_bsr_loss_weight = self._current_bsr_loss_weight()
                    loss = loss + current_bsr_loss_weight * bsr_total_loss
                    output.bsr_loss_terms = bsr_losses
                    output.bsr_loss_weight = torch.tensor(
                        current_bsr_loss_weight,
                        device=loss.device,
                        dtype=loss.dtype,
                    )
                    output.refine_loss = bsr_losses["refine_loss"]
            except Exception as e:
                log.warning(f"BSR loss computation failed: {e}")

        return loss, output

    def _set_log_batch_size(self, batch: Any) -> None:
        """Capture a stable batch size used by Lightning metric logging.

        Our batches are nested/heterogeneous (NAG/Data), so Lightning's
        automatic inference may pick the number of points instead of the
        number of graphs/scenes.
        """
        candidate = batch[0] if isinstance(batch, (list, tuple)) and len(batch) > 0 else batch
        batch_size = getattr(candidate, 'num_graphs', None)
        if batch_size is None:
            batch_size = 1
        self._log_batch_size = int(batch_size)

    def _batch_size_for_logging(self) -> int:
        return int(getattr(self, '_log_batch_size', 1))

    def step_single_run_inference(self, nag: NAG) -> SemanticSegmentationOutput:
        """Single-run inference
        """
        output = self.forward(nag)
        output = self.get_target(nag, output)
        return output

    def step_multi_run_inference(
            self,
            nag: NAG,
            transform: Transform,
            num_runs: int,
            key: str = 'tta_node_id'
    ) -> SemanticSegmentationOutput:
        """Multi-run inference, typically with test-time augmentation.
        See `BaseDataModule.on_after_batch_transfer`
        """
        # Since the transform may change the sampling of the nodes, we
        # save their input id here before anything. This will allow us
        # to fuse the multiple predictions for each node
        transform.transforms = [NAGSaveNodeIndex(key=key)] \
                               + transform.transforms

        # Create empty output predictions, to be iteratively populated
        # with the multiple predictions
        output_multi = self._create_empty_output(nag)

        # Recover the target labels from the reference NAG
        output_multi = self.get_target(nag, output_multi)

        # Build the global logits, in which the multi-run
        # logits will be accumulated, before computing their final
        seen = torch.zeros(nag.num_points[1], dtype=torch.bool)

        for i_run in range(num_runs):

            # Apply transform
            nag_ = transform(nag.clone())

            # Forward pass
            output = self.forward(nag_)

            # Update the output results
            output_multi = self._update_output_multi(
                output_multi, nag, output, nag_, key)

            # Maintain the seen/unseen mask for first segment-level nodes only
            node_id = nag_[1][key]
            seen[node_id] = True

        # Restore the original transform inplace modification
        transform.transforms = transform.transforms[1:]

        # If some nodes were not seen across any of the multi-runs,
        # search their nearest seen neighbor
        unseen_idx = torch.where(~seen)[0]
        batch = nag[1].batch
        if unseen_idx.shape[0] > 0:
            seen_idx = torch.where(seen)[0]
            x_search = nag[1].pos[seen_idx]
            x_query = nag[1].pos[unseen_idx]
            neighbors = knn_2(
                x_search,
                x_query,
                1,
                r_max=2,
                batch_search=batch[seen_idx] if batch is not None else None,
                batch_query=batch[unseen_idx] if batch is not None else None)[0]
            num_unseen = unseen_idx.shape[0]
            num_seen = seen_idx.shape[0]
            num_left_out = (neighbors == -1).sum().long()
            if num_left_out > 0:
                log.warning(
                    f"Could not find a neighbor for all unseen nodes: num_seen="
                    f"{num_seen}, num_unseen={num_unseen}, num_left_out="
                    f"{num_left_out}. These left out nodes will default to "
                    f"label-0 class prediction. Consider sampling less nodes "
                    f"in the augmentations, or increase the search radius")

            # Propagate the output to unseen neighbors
            output_multi = self._propagate_output_to_unseen_neighbors(
                output_multi, nag, seen, neighbors)

        return output_multi

    def _create_empty_output(self, nag: NAG) -> SemanticSegmentationOutput:
        """Local helper method to initialize an empty output for
        multi-run prediction.
        """
        device = nag.device
        num_classes = self.num_classes
        if self.multi_stage_loss:
            logits = [
                torch.zeros(num_points, num_classes, device=device)
                for num_points in nag.num_points[1:]]
        else:
            logits = torch.zeros(nag.num_points[1], num_classes, device=device)
        return SemanticSegmentationOutput(logits)

    @staticmethod
    def _update_output_multi(
            output_multi: SemanticSegmentationOutput,
            nag: NAG,
            output: SemanticSegmentationOutput,
            nag_transformed: NAG,
            key: str
    ) -> SemanticSegmentationOutput:
        """Local helper method to accumulate multiple predictions on
        the same--or part of the same--point cloud.
        """
        # Recover the node identifier that should have been
        # implanted by `NAGSaveNodeIndex` and forward on the
        # augmented data and update the global logits of the node
        if output.multi_stage:
            for i in range(len(output.logits)):
                node_id = nag_transformed[i + 1][key]
                output_multi.logits[i][node_id] += output.logits[i]
        else:
            node_id = nag_transformed[1][key]
            output_multi.logits[node_id] += output.logits
        return output_multi

    @staticmethod
    def _propagate_output_to_unseen_neighbors(
            output: SemanticSegmentationOutput,
            nag: NAG,
            seen: torch.Tensor,
            neighbors: torch.Tensor
    ) -> SemanticSegmentationOutput:
        """Local helper method to propagate predictions to unseen
        neighbors.
        """
        seen_idx = torch.where(seen)[0]
        unseen_idx = torch.where(~seen)[0]
        if output.multi_stage:
            output.logits[0][unseen_idx] = output.logits[0][seen_idx][neighbors]
        else:
            output.logits[unseen_idx] = output.logits[seen_idx][neighbors]
        return output

    def get_target(
            self,
            nag: NAG,
            output: SemanticSegmentationOutput
    ) -> SemanticSegmentationOutput:
        """Recover the target histogram of labels from the NAG object.
        The labels will be saved in `output.y_hist`.

        If the `multi_stage_loss=True`, a list of label histograms
        will be recovered (one for each prediction level).

        If `sampling_loss=True`, the histogram(s) will be updated based
        on the actual level-0 point sampling. That is, superpoints will
        be supervised by the labels of the sampled points at train time,
        rather than the true full-resolution label histogram.

        If no labels are found in the NAG, `output.y_hist` will be None.
        """
        assert not(self.hparams.sampling_loss and self.hparams.net.nano), \
            ("Sampling loss is not supported with the `nano`, as fast nano "
             "avoids loading the atom level, and sampling loss requires the "
             "atom level.")

        # Return if the required labels cannot be found in the NAG
        if self.hparams.sampling_loss and nag[0].y is None:
            output.y_hist = None
            return output
        elif self.multi_stage_loss:
            for i in range(1, nag.absolute_num_levels):
                if nag[i].y is None:
                    output.y_hist = None
                    return output
        elif nag[1].y is None:
            output.y_hist = None
            return output

        # Recover level-1 label histograms, either from the level-0
        # sampled points (i.e. sampling will affect the loss and metrics)
        # or directly from the precomputed level-1 label histograms (i.e.
        # true annotations)
        if self.hparams.sampling_loss and self.multi_stage_loss:
            y_hist = [
                atomic_to_histogram(
                    nag[0].y,
                    nag.get_super_index(i_level, low = 0),
                    n_bins=self.num_classes + 1)
                for i_level in range(1, nag.num_levels)]

        elif self.hparams.sampling_loss:
            idx = nag[0].super_index
            y = nag[0].y

            # Convert level-0 labels to segment-level histograms, while
            # accounting for the extra class for unlabeled/ignored points
            y_hist = atomic_to_histogram(y, idx, n_bins=self.num_classes + 1)

        elif self.multi_stage_loss:
            y_hist = [nag[i_level].y for i_level in range(1, nag.num_levels)]

        else:
            y_hist = nag[1].y

        # Store the label histogram in the output object
        output.y_hist = y_hist
        if getattr(nag[0], 'super_index', None) is not None:
            output.super_index = nag[0].super_index
        if getattr(nag[0], 'y', None) is not None:
            output.point_y = nag[0].y

        return output

    def training_step(
            self,
            batch: NAG,
            batch_idx: int
    ) -> torch.Tensor:
        self._set_log_batch_size(batch)
        loss, output = self.model_step(batch)
        if not torch.isfinite(loss):
            log.warning(
                "Non-finite training loss detected at batch %s; "
                "skipping optimizer update for this batch.",
                batch_idx,
            )
            loss = torch.zeros(
                (),
                device=loss.device,
                dtype=loss.dtype,
                requires_grad=True,
            )

        # Update and log metrics
        self.train_step_update_metrics(loss, output)
        self.train_step_log_metrics()

        # Explicitly delete the output, for memory release
        del output

        # return loss or backpropagation will fail
        return loss

    def train_step_update_metrics(
            self,
            loss: torch.Tensor,
            output: SemanticSegmentationOutput
    ) -> None:
        """Update train metrics after a single step, with the content of
        the output object.
        """
        self.train_loss(loss.detach())
        self._update_semantic_confusion_matrix(self.train_cm, output)
        if self.bsr_enabled:
            zero = torch.tensor(0.0, device=loss.device)
            loss_terms = getattr(output, 'bsr_loss_terms', None)
            bsr_output = getattr(output, 'bsr_output', None)
            self.train_refine_loss(loss_terms["refine_loss"].detach() if loss_terms else zero)
            self.train_consistency_loss(loss_terms["consistency_loss"].detach() if loss_terms else zero)
            self.train_diversity_loss(loss_terms["diversity_loss"].detach() if loss_terms else zero)
            self.train_boundary_loss(loss_terms["boundary_loss"].detach() if loss_terms else zero)
            if bsr_output is not None and bsr_output.num_superpoints > 0:
                mean_score = (
                    bsr_output.candidate_scores.detach().mean()
                    if bsr_output.candidate_scores.numel() > 0 else zero
                )
                sample_valid_ratio = (
                    bsr_output.sampled_point_mask.detach().float().mean()
                    if bsr_output.sampled_point_mask is not None
                    and bsr_output.sampled_point_mask.numel() > 0 else zero
                )
                self.train_candidate_ratio(torch.tensor(bsr_output.candidate_ratio, device=loss.device))
                self.train_candidate_score(mean_score)
                self.train_sample_valid_ratio(sample_valid_ratio)
                self.train_num_valid_sampled_points(
                    torch.tensor(float(bsr_output.num_valid_sampled_points), device=loss.device))
                self.train_avg_valid_points_per_candidate(
                    torch.tensor(bsr_output.avg_valid_points_per_candidate, device=loss.device))
                self.train_effective_refine_ratio(
                    torch.tensor(bsr_output.effective_refine_ratio, device=loss.device))
                self.train_point_gate_mean(
                    self._current_bsr_point_gate_mean(bsr_output, loss.device))
                self.train_assignment_entropy(
                    self._current_bsr_output_mean(bsr_output, 'assignment_entropy', loss.device))
                self.train_secondary_slot_mass(
                    self._current_bsr_output_mean(bsr_output, 'secondary_slot_mass', loss.device))
                self.train_dual_slot_activation_ratio(
                    self._current_bsr_output_mean(bsr_output, 'dual_slot_activation_ratio', loss.device))
                self.train_slot_diversity(
                    self._current_bsr_output_mean(bsr_output, 'slot_diversity', loss.device))
                self.train_point_propagation_coverage(
                    torch.tensor(
                        float(getattr(bsr_output, 'point_propagation_coverage', 0.0)),
                        device=loss.device))
                selector_terms = getattr(bsr_output, 'selector_score_terms', {}) or {}
                selector_summary = getattr(bsr_output, 'selector_score_summary', {}) or {}
                for term, metric in self.train_bsr_selector_terms.items():
                    term_values = selector_terms.get(term)
                    metric(term_values.detach().mean() if term_values is not None and term_values.numel() > 0 else zero)
                for term, metric in self.train_bsr_selector_term_vars.items():
                    term_summary = selector_summary.get(term, {})
                    metric(torch.tensor(term_summary.get("var", 0.0), device=loss.device))
            else:
                self.train_candidate_ratio(zero)
                self.train_candidate_score(zero)
                self.train_sample_valid_ratio(zero)
                self.train_num_valid_sampled_points(zero)
                self.train_avg_valid_points_per_candidate(zero)
                self.train_effective_refine_ratio(zero)
                self.train_point_gate_mean(zero)
                self.train_assignment_entropy(zero)
                self.train_secondary_slot_mass(zero)
                self.train_dual_slot_activation_ratio(zero)
                self.train_slot_diversity(zero)
                self.train_point_propagation_coverage(zero)
                for metric in self.train_bsr_selector_terms.values():
                    metric(zero)
                for metric in self.train_bsr_selector_term_vars.values():
                    metric(zero)
    def train_step_log_metrics(self) -> None:
        """Log train metrics after a single step with the content of the
        output object.
        """
        batch_size = self._batch_size_for_logging()
        self.log(
            "train/loss",
            self.train_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=batch_size)
        if self.bsr_enabled:
            self.log(
                "train/bsr_refine_loss",
                self.train_refine_loss,
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                batch_size=batch_size)
            self.log(
                "train/bsr_consistency_loss",
                self.train_consistency_loss,
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                batch_size=batch_size)
            self.log(
                "train/bsr_diversity_loss",
                self.train_diversity_loss,
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                batch_size=batch_size)
            self.log(
                "train/bsr_boundary_loss",
                self.train_boundary_loss,
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                batch_size=batch_size)
            self.log(
                "train/bsr_candidate_ratio",
                self.train_candidate_ratio,
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                batch_size=batch_size)
            self.log(
                "train/bsr_candidate_score",
                self.train_candidate_score,
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                batch_size=batch_size)
            self.log(
                "train/bsr_sample_valid_ratio",
                self.train_sample_valid_ratio,
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                batch_size=batch_size)
            self.log(
                "train/bsr_num_valid_sampled_points",
                self.train_num_valid_sampled_points,
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                batch_size=batch_size)
            self.log(
                "train/bsr_avg_valid_points_per_candidate",
                self.train_avg_valid_points_per_candidate,
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                batch_size=batch_size)
            self.log(
                "train/bsr_effective_refine_ratio",
                self.train_effective_refine_ratio,
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                batch_size=batch_size)
            self.log(
                "train/bsr_point_gate_mean",
                self.train_point_gate_mean,
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                batch_size=batch_size)
            self.log(
                "train/bsr_assignment_entropy",
                self.train_assignment_entropy,
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                batch_size=batch_size)
            self.log(
                "train/bsr_secondary_slot_mass",
                self.train_secondary_slot_mass,
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                batch_size=batch_size)
            self.log(
                "train/bsr_dual_slot_activation_ratio",
                self.train_dual_slot_activation_ratio,
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                batch_size=batch_size)
            self.log(
                "train/bsr_slot_diversity",
                self.train_slot_diversity,
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                batch_size=batch_size)
            self.log(
                "train/bsr_point_propagation_coverage",
                self.train_point_propagation_coverage,
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                batch_size=batch_size)
            for term, metric in self.train_bsr_selector_terms.items():
                self.log(
                    f"train/bsr_selector_{term}",
                    metric,
                    on_step=False,
                    on_epoch=True,
                    prog_bar=False,
                    batch_size=batch_size)
            for term, metric in self.train_bsr_selector_term_vars.items():
                self.log(
                    f"train/bsr_selector_{term}_var",
                    metric,
                    on_step=False,
                    on_epoch=True,
                    prog_bar=False,
                    batch_size=batch_size)
            self.log(
                "train/bsr_aux_weight",
                self._current_bsr_loss_weight(),
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                batch_size=batch_size)
            self.log(
                "train/bsr_fusion_alpha",
                self._current_bsr_fusion_alpha(),
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                batch_size=batch_size)
            train_total = self._bsr_stage_total_count.get('train', 0)
            if train_total > 0:
                fail_rate = self._bsr_stage_fail_count.get('train', 0) / train_total
                self.log(
                    "train/bsr_fail_rate",
                    fail_rate,
                    on_step=False,
                    on_epoch=True,
                    prog_bar=False,
                    batch_size=batch_size)
    def on_train_epoch_end(self) -> None:
        self._on_train_epoch_end(
            cm=self.train_cm,
            metric_category='superpoints_semantic_prediction')

    def _on_train_epoch_end(
            self,
            cm: ConfusionMatrix,
            metric_category: str) -> None:

        # Retrieving the appropriate prefix for logging metrics
        if metric_category == 'superpoints_semantic_prediction':
            prefix = ''
        elif metric_category == 'superpoints_purity':
            prefix = 'o'
        elif metric_category == 'edge_classification':
            prefix = 'edge_classification_'
        else:
            raise ValueError(
                f"Invalid metric_category: {metric_category}\n"
                "Valid values are : superpoints_semantic_prediction, "
                "superpoints_purity, edge_classification")

        if self.trainer.num_devices > 1:
            epoch_cm = torch.sum(self.all_gather(cm.confmat), dim=0)
            epoch_cm = ConfusionMatrix(
                self.num_classes).from_confusion_matrix(epoch_cm)
        else:
            epoch_cm = cm

        # Log metrics
        self.log(
            f"train/{prefix}miou",
            epoch_cm.miou(),
            prog_bar=True,
            rank_zero_only=True)
        self.log(
            f"train/{prefix}oa",
            epoch_cm.oa(),
            prog_bar=True,
            rank_zero_only=True)
        self.log(
            f"train/{prefix}macc",
            epoch_cm.macc(),
            prog_bar=True,
            rank_zero_only=True)

        if getattr(self.hparams, 'extensive_logging', True):
            class_names = (
                self.class_names if prefix != 'edge_classification_'
                else ['inter', 'intra'])
            for iou, seen, name in zip(*epoch_cm.iou(), class_names):
                if seen:
                    self.log(
                        f"train/{prefix}iou_{name}", iou, prog_bar=True,
                        rank_zero_only=True)

            if prefix == 'edge_classification_':
                inter_index = 0

                num_inter_target = (
                    epoch_cm.confmat[inter_index, inter_index] +
                    epoch_cm.confmat[inter_index, 1-inter_index])
                num_inter_pred = (
                    epoch_cm.confmat[inter_index, inter_index] +
                    epoch_cm.confmat[1-inter_index, inter_index])

                inter_recall = (
                    epoch_cm.confmat[inter_index, inter_index] /
                    num_inter_target)
                inter_precision = (
                    epoch_cm.confmat[inter_index, inter_index] /
                    num_inter_pred)
                inter_f1 = (
                    2 * inter_precision * inter_recall /
                    (inter_precision + inter_recall))
                inter_prediction_rate = (
                    num_inter_pred / (epoch_cm.confmat[:2, :2].sum()))

                self.log(
                    f"train/{prefix}inter_recall", inter_recall*100,
                    prog_bar=True, rank_zero_only=True)
                self.log(
                    f"train/{prefix}inter_precision", inter_precision*100,
                    prog_bar=True, rank_zero_only=True)
                self.log(
                    f"train/{prefix}inter_f1", inter_f1*100,
                    prog_bar=True, rank_zero_only=True)
                self.log(
                    f"train/{prefix}inter_prediction_rate",
                    inter_prediction_rate*100, prog_bar=True,
                    rank_zero_only=True)

        # Reset metrics accumulated over the last epoch
        cm.reset()
        epoch_cm.reset()

    def validation_step(
            self,
            batch: NAG,
            batch_idx: int
    ) -> None:
        self._set_log_batch_size(batch)
        loss, output = self.model_step(batch)

        # Update and log metrics
        self.validation_step_update_metrics(loss, output)
        self.validation_step_log_metrics()

        # Get the current epoch. For the validation set, we alter the
        # epoch number so that `track_val_every_n_epoch` can align
        # with `check_val_every_n_epoch`. Indeed, it seems the epoch
        # number during the validation step is always one increment
        # ahead
        epoch = self.current_epoch + 1

        # Store features and predictions for a batch of interest
        # NB: the `batch_idx` produced by torch lightning here
        # corresponds to the `Dataloader`'s index wrt the current epoch
        # and NOT an index wrt the `Dataset`. Said otherwise, if the
        # `Dataloader(shuffle=True)` then, the stored batch will not be
        # the same at each epoch. For this reason, if tracking the same
        # object across training is needed, the `Dataloader` and the
        # transforms should be free from any stochasticity
        track_epoch = epoch % self.hparams.track_val_every_n_epoch == 0
        track_batch = batch_idx == self.hparams.track_val_idx
        track_all_batches = self.hparams.track_val_idx == -1
        if track_epoch and (track_batch or track_all_batches):
            self.track_batch(batch, batch_idx, output)

        # Explicitly delete the output, for memory release
        del output

    def validation_step_update_metrics(
            self,
            loss: torch.Tensor,
            output: SemanticSegmentationOutput
    ) -> None:
        """Update validation metrics with the content of the output
        object.
        """
        self.val_loss(loss.detach())
        self._update_semantic_confusion_matrix(self.val_cm, output)
        if self.bsr_enabled:
            zero = torch.tensor(0.0, device=loss.device)
            loss_terms = getattr(output, 'bsr_loss_terms', None)
            bsr_output = getattr(output, 'bsr_output', None)
            self.val_refine_loss(loss_terms["refine_loss"].detach() if loss_terms else zero)
            self.val_consistency_loss(loss_terms["consistency_loss"].detach() if loss_terms else zero)
            self.val_diversity_loss(loss_terms["diversity_loss"].detach() if loss_terms else zero)
            self.val_boundary_loss(loss_terms["boundary_loss"].detach() if loss_terms else zero)
            if bsr_output is not None and bsr_output.num_superpoints > 0:
                mean_score = (
                    bsr_output.candidate_scores.detach().mean()
                    if bsr_output.candidate_scores.numel() > 0 else zero
                )
                sample_valid_ratio = (
                    bsr_output.sampled_point_mask.detach().float().mean()
                    if bsr_output.sampled_point_mask is not None
                    and bsr_output.sampled_point_mask.numel() > 0 else zero
                )
                self.val_candidate_ratio(torch.tensor(bsr_output.candidate_ratio, device=loss.device))
                self.val_candidate_score(mean_score)
                self.val_sample_valid_ratio(sample_valid_ratio)
                self.val_num_valid_sampled_points(
                    torch.tensor(float(bsr_output.num_valid_sampled_points), device=loss.device))
                self.val_avg_valid_points_per_candidate(
                    torch.tensor(bsr_output.avg_valid_points_per_candidate, device=loss.device))
                self.val_effective_refine_ratio(
                    torch.tensor(bsr_output.effective_refine_ratio, device=loss.device))
                self.val_point_gate_mean(
                    self._current_bsr_point_gate_mean(bsr_output, loss.device))
                self.val_assignment_entropy(
                    self._current_bsr_output_mean(bsr_output, 'assignment_entropy', loss.device))
                self.val_secondary_slot_mass(
                    self._current_bsr_output_mean(bsr_output, 'secondary_slot_mass', loss.device))
                self.val_dual_slot_activation_ratio(
                    self._current_bsr_output_mean(bsr_output, 'dual_slot_activation_ratio', loss.device))
                self.val_slot_diversity(
                    self._current_bsr_output_mean(bsr_output, 'slot_diversity', loss.device))
                self.val_point_propagation_coverage(
                    torch.tensor(
                        float(getattr(bsr_output, 'point_propagation_coverage', 0.0)),
                        device=loss.device))
                selector_terms = getattr(bsr_output, 'selector_score_terms', {}) or {}
                selector_summary = getattr(bsr_output, 'selector_score_summary', {}) or {}
                for term, metric in self.val_bsr_selector_terms.items():
                    term_values = selector_terms.get(term)
                    metric(term_values.detach().mean() if term_values is not None and term_values.numel() > 0 else zero)
                for term, metric in self.val_bsr_selector_term_vars.items():
                    term_summary = selector_summary.get(term, {})
                    metric(torch.tensor(term_summary.get("var", 0.0), device=loss.device))
            else:
                self.val_candidate_ratio(zero)
                self.val_candidate_score(zero)
                self.val_sample_valid_ratio(zero)
                self.val_num_valid_sampled_points(zero)
                self.val_avg_valid_points_per_candidate(zero)
                self.val_effective_refine_ratio(zero)
                self.val_point_gate_mean(zero)
                self.val_assignment_entropy(zero)
                self.val_secondary_slot_mass(zero)
                self.val_dual_slot_activation_ratio(zero)
                self.val_slot_diversity(zero)
                self.val_point_propagation_coverage(zero)
                for metric in self.val_bsr_selector_terms.values():
                    metric(zero)
                for metric in self.val_bsr_selector_term_vars.values():
                    metric(zero)
    def validation_step_log_metrics(self) -> None:
        """Log validation metrics after a single step with the content
        of the output object.
        """
        batch_size = self._batch_size_for_logging()
        self.log(
            "val/loss", self.val_loss, on_step=False, on_epoch=True,
            prog_bar=True, batch_size=batch_size)
        if self.bsr_enabled:
            self.log(
                "val/bsr_refine_loss",
                self.val_refine_loss,
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                batch_size=batch_size)
            self.log(
                "val/bsr_consistency_loss",
                self.val_consistency_loss,
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                batch_size=batch_size)
            self.log(
                "val/bsr_diversity_loss",
                self.val_diversity_loss,
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                batch_size=batch_size)
            self.log(
                "val/bsr_boundary_loss",
                self.val_boundary_loss,
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                batch_size=batch_size)
            self.log(
                "val/bsr_candidate_ratio",
                self.val_candidate_ratio,
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                batch_size=batch_size)
            self.log(
                "val/bsr_candidate_score",
                self.val_candidate_score,
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                batch_size=batch_size)
            self.log(
                "val/bsr_sample_valid_ratio",
                self.val_sample_valid_ratio,
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                batch_size=batch_size)
            self.log(
                "val/bsr_num_valid_sampled_points",
                self.val_num_valid_sampled_points,
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                batch_size=batch_size)
            self.log(
                "val/bsr_avg_valid_points_per_candidate",
                self.val_avg_valid_points_per_candidate,
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                batch_size=batch_size)
            self.log(
                "val/bsr_effective_refine_ratio",
                self.val_effective_refine_ratio,
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                batch_size=batch_size)
            self.log(
                "val/bsr_point_gate_mean",
                self.val_point_gate_mean,
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                batch_size=batch_size)
            self.log(
                "val/bsr_assignment_entropy",
                self.val_assignment_entropy,
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                batch_size=batch_size)
            self.log(
                "val/bsr_secondary_slot_mass",
                self.val_secondary_slot_mass,
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                batch_size=batch_size)
            self.log(
                "val/bsr_dual_slot_activation_ratio",
                self.val_dual_slot_activation_ratio,
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                batch_size=batch_size)
            self.log(
                "val/bsr_slot_diversity",
                self.val_slot_diversity,
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                batch_size=batch_size)
            self.log(
                "val/bsr_point_propagation_coverage",
                self.val_point_propagation_coverage,
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                batch_size=batch_size)
            for term, metric in self.val_bsr_selector_terms.items():
                self.log(
                    f"val/bsr_selector_{term}",
                    metric,
                    on_step=False,
                    on_epoch=True,
                    prog_bar=False,
                    batch_size=batch_size)
            for term, metric in self.val_bsr_selector_term_vars.items():
                self.log(
                    f"val/bsr_selector_{term}_var",
                    metric,
                    on_step=False,
                    on_epoch=True,
                    prog_bar=False,
                    batch_size=batch_size)
            self.log(
                "val/bsr_aux_weight",
                self._current_bsr_loss_weight(),
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                batch_size=batch_size)
            self.log(
                "val/bsr_fusion_alpha",
                self._current_bsr_fusion_alpha(),
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                batch_size=batch_size)
            val_total = self._bsr_stage_total_count.get('val', 0)
            if val_total > 0:
                self.log(
                    "val/bsr_fail_rate",
                    self._bsr_stage_fail_count.get('val', 0) / val_total,
                    on_step=False,
                    on_epoch=True,
                    prog_bar=False,
                    batch_size=batch_size)
    def _on_eval_epoch_end(
            self,
            stage: str,
            cm: ConfusionMatrix,
            metric_category: str,
            miou_best: MaxMetric = None,
            oa_best: MaxMetric = None,
            macc_best: MaxMetric = None,
    ) -> None:
        """Helper method to factorize validation and test epoch end logic.

        :param stage: str
            The current stage. It is used to know under which name
            the metrics should be logged, and also to log `test` specific 
            metrics.

            The values must be one of the following:
            - 'val' for validation epochs,
            - 'test' for test epochs.

        :param cm: ConfusionMatrix to compute metrics from

        :param metric_category: str
            The type of metrics to compute.
            The values must be one of the following:
            - 'superpoints_semantic_prediction' for semantic prediction of 
            superpoints
            - 'superpoints_purity' for purity of superpoints
            - 'edge_classification' for edge classification
            This is used to determine the correct name for logging metrics.
            (should it be `omiou` or `miou`, etc.)

        :param miou_best: MaxMetric
            Metric tracking best mIoU
            Only relevant and used for validation.
        :param oa_best: MaxMetric
            Metric tracking best OA
            Only relevant and used for validation.
        :param macc_best: MaxMetric
            Metric tracking best mAcc
            Only relevant and used for validation.

        """
        assert stage in ['val', 'test']

        if metric_category == 'superpoints_semantic_prediction':
            prefix = ''
        elif metric_category == 'superpoints_purity':
            prefix = 'o'
        else:
            raise ValueError(
                f"Invalid metric_category: {metric_category}\n"
                f"Valid values are : superpoints_semantic_prediction, "
                f"superpoints_purity")

        if stage == 'val' :
            assert miou_best is not None
            assert oa_best is not None
            assert macc_best is not None

        if stage == 'test':
            # Finalize the submission
            if self.trainer.datamodule.hparams.submit:
                self.trainer.datamodule.test_dataset.finalize_submission(
                    self.submission_dir)

            if not self.test_has_target:
                cm.reset()
                return

        if self.trainer.num_devices > 1:
            epoch_cm = torch.sum(self.all_gather(cm.confmat), dim=0)
            epoch_cm = ConfusionMatrix(
                self.num_classes).from_confusion_matrix(epoch_cm)
        else:
            epoch_cm = cm

        miou = epoch_cm.miou()
        oa = epoch_cm.oa()
        macc = epoch_cm.macc()

        # Log metrics
        self.log(
            f"{stage}/{prefix}miou",
            miou,
            prog_bar=True,
            rank_zero_only=True)
        self.log(
            f"{stage}/{prefix}oa",
            oa,
            prog_bar=True,
            rank_zero_only=True)
        self.log(
            f"{stage}/{prefix}macc",
            macc,
            prog_bar=True,
            rank_zero_only=True)

        class_names = self.class_names
        for iou, seen, name in zip(*epoch_cm.iou(), class_names):
            if seen:
                self.log(
                    f"{stage}/{prefix}iou_{name}",
                    iou,
                    prog_bar=True,
                    rank_zero_only=True)

        if stage == 'val' :
            best_epoch_key = f"val/{prefix}miou_best_epoch"
            if best_epoch_key in self._val_best_epochs:
                previous_miou_best = _scalar_float(
                    miou_best.compute(), default=float('-inf'))
            else:
                previous_miou_best = float('-inf')
            miou_value = _scalar_float(miou, default=float('-inf'))
            miou_improved = miou_value >= previous_miou_best

            # Update best-so-far metrics
            miou_best(miou)
            oa_best(oa)
            macc_best(macc)
            current_miou_best = miou_best.compute()

            # Log best-so-far metrics, using `.compute()` instead of passing
            # the whole torchmetrics object, because otherwise metric would
            # be reset by lightning after each epoch
            self.log(
                f"val/{prefix}miou_best",
                current_miou_best,
                prog_bar=True,
                rank_zero_only=True)
            self.log(
                f"val/{prefix}oa_best",
                oa_best.compute(),
                prog_bar=True,
                rank_zero_only=True)
            self.log(
                f"val/{prefix}macc_best",
                macc_best.compute(),
                prog_bar=True,
                rank_zero_only=True)
            self.log(
                f"val/{prefix}miou_gap_to_best",
                current_miou_best - miou,
                prog_bar=False,
                rank_zero_only=True)

            if miou_improved:
                self._val_best_epochs[best_epoch_key] = int(self.current_epoch)
            best_epoch = self._val_best_epochs.get(best_epoch_key)
            if best_epoch is not None:
                self.log(
                    best_epoch_key,
                    float(best_epoch),
                    prog_bar=False,
                    rank_zero_only=True)
                for logger in self._iter_wandb_loggers():
                    try:
                        logger.experiment.summary[best_epoch_key] = best_epoch
                    except Exception as exc:
                        log.warning("Could not update W&B best epoch summary: %s", exc)

        elif getattr(self.hparams, 'extensive_logging', True) and stage == 'test':
            # Log confusion matrix to wandb
            if isinstance(self.logger, WandbLogger):
                self.logger.experiment.log({
                    f"test/{prefix}cm": wandb_confusion_matrix(
                        epoch_cm.confmat, class_names=self.class_names)})

        # Reset metrics accumulated over the last epoch
        cm.reset()
        epoch_cm.reset()

    def on_validation_epoch_end(self) -> None:
        self._on_eval_epoch_end(
            stage='val',
            cm=self.val_cm,
            metric_category='superpoints_semantic_prediction',
            miou_best=self.val_miou_best,
            oa_best=self.val_oa_best,
            macc_best=self.val_macc_best)

    def on_test_start(self) -> None:
        # Initialize the submission directory based on the time of the
        # beginning of test. This way, the test steps can all have
        # access to the same directory, regardless of their execution
        # time
        self.submission_dir = self.trainer.datamodule.test_dataset.submission_dir
        self.on_fit_start()

    def test_step(self, batch: NAG, batch_idx: int) -> None:
        self._set_log_batch_size(batch)
        loss, output = self.model_step(batch)

        # If the input batch does not have any labels (e.g. test set
        # with held-out labels), y_hist will be None and the loss will
        # not be computed. In this case, we arbitrarily set the loss to
        # 0 and do not update the confusion matrix
        loss = 0 if loss is None else loss

        # If the test set misses targets, we keep track of it, to skip
        # metrics computation on the test set
        if not output.has_target:
            self.test_has_target = False

        # Update and log metrics
        self.test_step_update_metrics(loss, output)
        self.test_step_log_metrics()

        # Prepare submission for held-out test sets
        if self.trainer.datamodule.hparams.submit:
            nag = batch if isinstance(batch, NAG) else batch[0]
            l0_pos = nag[0].pos.detach().cpu()
            l0_pred = output.point_semantic_pred(super_index=nag[0].super_index).detach().cpu()
            self.trainer.datamodule.test_dataset.make_submission(
                batch_idx, l0_pred, l0_pos, submission_dir=self.submission_dir)

        # Store features and predictions for a batch of interest
        # NB: the `batch_idx` produced by torch lightning here
        # corresponds to the `Dataloader`'s index wrt the current epoch
        # and NOT an index wrt the `Dataset`. Said otherwise, if the
        # `Dataloader(shuffle=True)` then, the stored batch will not be
        # the same at each epoch. For this reason, if tracking the same
        # object across training is needed, the `Dataloader` and the
        # transforms should be free from any stochasticity
        track_batch = batch_idx == self.hparams.track_test_idx
        track_all_batches = self.hparams.track_test_idx == -1
        if track_batch or track_all_batches:
            self.track_batch(batch, batch_idx, output)

        # Explicitly delete the output, for memory release
        del output

    def test_step_update_metrics(
            self,
            loss: torch.Tensor,
            output: SemanticSegmentationOutput
    ) -> None:
        """Update test metrics with the content of the output object.
        """
        # If the test set misses targets, we keep track of it, to skip
        # metrics computation on the test set
        if not self.test_has_target:
            return

        self.test_loss(loss.detach())
        self._update_semantic_confusion_matrix(self.test_cm, output)
        if self.bsr_enabled:
            zero = torch.tensor(0.0, device=loss.device)
            loss_terms = getattr(output, 'bsr_loss_terms', None)
            bsr_output = getattr(output, 'bsr_output', None)
            self.test_refine_loss(loss_terms["refine_loss"].detach() if loss_terms else zero)
            self.test_consistency_loss(loss_terms["consistency_loss"].detach() if loss_terms else zero)
            self.test_diversity_loss(loss_terms["diversity_loss"].detach() if loss_terms else zero)
            self.test_boundary_loss(loss_terms["boundary_loss"].detach() if loss_terms else zero)
            if bsr_output is not None and bsr_output.num_superpoints > 0:
                mean_score = (
                    bsr_output.candidate_scores.detach().mean()
                    if bsr_output.candidate_scores.numel() > 0 else zero
                )
                sample_valid_ratio = (
                    bsr_output.sampled_point_mask.detach().float().mean()
                    if bsr_output.sampled_point_mask is not None
                    and bsr_output.sampled_point_mask.numel() > 0 else zero
                )
                self.test_candidate_ratio(torch.tensor(bsr_output.candidate_ratio, device=loss.device))
                self.test_candidate_score(mean_score)
                self.test_sample_valid_ratio(sample_valid_ratio)
                self.test_num_valid_sampled_points(
                    torch.tensor(float(bsr_output.num_valid_sampled_points), device=loss.device))
                self.test_avg_valid_points_per_candidate(
                    torch.tensor(bsr_output.avg_valid_points_per_candidate, device=loss.device))
                self.test_effective_refine_ratio(
                    torch.tensor(bsr_output.effective_refine_ratio, device=loss.device))
                self.test_point_gate_mean(
                    self._current_bsr_point_gate_mean(bsr_output, loss.device))
                self.test_assignment_entropy(
                    self._current_bsr_output_mean(bsr_output, 'assignment_entropy', loss.device))
                self.test_secondary_slot_mass(
                    self._current_bsr_output_mean(bsr_output, 'secondary_slot_mass', loss.device))
                self.test_dual_slot_activation_ratio(
                    self._current_bsr_output_mean(bsr_output, 'dual_slot_activation_ratio', loss.device))
                self.test_slot_diversity(
                    self._current_bsr_output_mean(bsr_output, 'slot_diversity', loss.device))
                self.test_point_propagation_coverage(
                    torch.tensor(
                        float(getattr(bsr_output, 'point_propagation_coverage', 0.0)),
                        device=loss.device))
                selector_terms = getattr(bsr_output, 'selector_score_terms', {}) or {}
                selector_summary = getattr(bsr_output, 'selector_score_summary', {}) or {}
                for term, metric in self.test_bsr_selector_terms.items():
                    term_values = selector_terms.get(term)
                    metric(term_values.detach().mean() if term_values is not None and term_values.numel() > 0 else zero)
                for term, metric in self.test_bsr_selector_term_vars.items():
                    term_summary = selector_summary.get(term, {})
                    metric(torch.tensor(term_summary.get("var", 0.0), device=loss.device))
            else:
                self.test_candidate_ratio(zero)
                self.test_candidate_score(zero)
                self.test_sample_valid_ratio(zero)
                self.test_num_valid_sampled_points(zero)
                self.test_avg_valid_points_per_candidate(zero)
                self.test_effective_refine_ratio(zero)
                self.test_point_gate_mean(zero)
                self.test_assignment_entropy(zero)
                self.test_secondary_slot_mass(zero)
                self.test_dual_slot_activation_ratio(zero)
                self.test_slot_diversity(zero)
                self.test_point_propagation_coverage(zero)
                for metric in self.test_bsr_selector_terms.values():
                    metric(zero)
                for metric in self.test_bsr_selector_term_vars.values():
                    metric(zero)
    def test_step_log_metrics(self) -> None:
        """Log test metrics after a single step with the content of the
        output object.
        """
        # If the test set misses targets, we keep track of it, to skip
        # metrics computation on the test set
        if not self.test_has_target:
            return
        
        batch_size = self._batch_size_for_logging()
        # As we don't prepare the train datasets, we cannot call
        # train_dataset.get_class_weight(), so the loss computation without
        # the proper weights is not possible.
        if not self.trainer.datamodule.hparams.prepare_only_test:
            self.log(
                "test/loss", self.test_loss, on_step=False, on_epoch=True,
                prog_bar=True, batch_size=batch_size)

        if self.bsr_enabled:
            self.log(
                "test/bsr_refine_loss",
                self.test_refine_loss,
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                batch_size=batch_size)
            self.log(
                "test/bsr_consistency_loss",
                self.test_consistency_loss,
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                batch_size=batch_size)
            self.log(
                "test/bsr_diversity_loss",
                self.test_diversity_loss,
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                batch_size=batch_size)
            self.log(
                "test/bsr_boundary_loss",
                self.test_boundary_loss,
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                batch_size=batch_size)
            self.log(
                "test/bsr_candidate_ratio",
                self.test_candidate_ratio,
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                batch_size=batch_size)
            self.log(
                "test/bsr_candidate_score",
                self.test_candidate_score,
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                batch_size=batch_size)
            self.log(
                "test/bsr_sample_valid_ratio",
                self.test_sample_valid_ratio,
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                batch_size=batch_size)
            self.log(
                "test/bsr_num_valid_sampled_points",
                self.test_num_valid_sampled_points,
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                batch_size=batch_size)
            self.log(
                "test/bsr_avg_valid_points_per_candidate",
                self.test_avg_valid_points_per_candidate,
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                batch_size=batch_size)
            self.log(
                "test/bsr_effective_refine_ratio",
                self.test_effective_refine_ratio,
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                batch_size=batch_size)
            self.log(
                "test/bsr_point_gate_mean",
                self.test_point_gate_mean,
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                batch_size=batch_size)
            self.log(
                "test/bsr_assignment_entropy",
                self.test_assignment_entropy,
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                batch_size=batch_size)
            self.log(
                "test/bsr_secondary_slot_mass",
                self.test_secondary_slot_mass,
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                batch_size=batch_size)
            self.log(
                "test/bsr_dual_slot_activation_ratio",
                self.test_dual_slot_activation_ratio,
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                batch_size=batch_size)
            self.log(
                "test/bsr_slot_diversity",
                self.test_slot_diversity,
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                batch_size=batch_size)
            self.log(
                "test/bsr_point_propagation_coverage",
                self.test_point_propagation_coverage,
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                batch_size=batch_size)
            for term, metric in self.test_bsr_selector_terms.items():
                self.log(
                    f"test/bsr_selector_{term}",
                    metric,
                    on_step=False,
                    on_epoch=True,
                    prog_bar=False,
                    batch_size=batch_size)
            for term, metric in self.test_bsr_selector_term_vars.items():
                self.log(
                    f"test/bsr_selector_{term}_var",
                    metric,
                    on_step=False,
                    on_epoch=True,
                    prog_bar=False,
                    batch_size=batch_size)
            self.log(
                "test/bsr_aux_weight",
                self._current_bsr_loss_weight(),
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                batch_size=batch_size)
            self.log(
                "test/bsr_fusion_alpha",
                self._current_bsr_fusion_alpha(),
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                batch_size=batch_size)
            test_total = self._bsr_stage_total_count.get('test', 0)
            if test_total > 0:
                self.log(
                    "test/bsr_fail_rate",
                    self._bsr_stage_fail_count.get('test', 0) / test_total,
                    on_step=False,
                    on_epoch=True,
                    prog_bar=False,
                    batch_size=batch_size)
    def on_test_epoch_end(self) -> None:
        self._on_eval_epoch_end(
            stage='test',
            cm=self.test_cm,
            metric_category='superpoints_semantic_prediction')

    def predict_step(
            self,
            batch: NAG,
            batch_idx: int
    ) -> Tuple[NAG, SemanticSegmentationOutput]:
        _, output = self.model_step(batch)
        return batch, output

    def track_batch(
            self,
            batch: NAG,
            batch_idx: int,
            output: SemanticSegmentationOutput,
            folder: str = None
    ) -> None:
        """Store a batch prediction to disk. The corresponding `NAG`
        object will be populated with semantic segmentation predictions
        for:
        - levels 1+ if `multi_stage` output (i.e. loss supervision on
          levels 1 and above)
        - only level 1 otherwise

        Besides, we also pre-compute the level-0 predictions as this is
        frequently required for downstream tasks. However, we choose not
        to compute the full-resolution predictions for the sake of disk
        memory.

        If a `folder` is provided, the NAG will be saved there under:
          <folder>/predictions/<stage>/<epoch>/batch_<batch_idx>.h5
        If not, the folder will be the logger's directory, if any.
        If not, the current working directory will be used.

        :param batch: NAG
            Object that will be stored to disk. Before that, the
            model predictions will be added to the attributes of each
            level, to facilitate downstream use of the stored `NAG`
        :param batch_idx: int
            Index of the batch to be stored
        :param output: SemanticSegmentationOutput
             Output of `self.model_step()`
        :param folder: str
            Path where to save the tracked batch. If not provided, the
            logger's saving directory will be used as fallback. If not
            logger is found, the current working directory will be used
        :return:
        """
        # Sanity check in case using multi-run inference
        if not isinstance(batch, NAG):
            raise NotImplementedError(
                f"Expected as NAG, but received a {type(batch)}. Are you "
                f"perhaps running multi-run inference ? If so, this is not "
                f"compatible with batch_saving, please deactivate either one.")

        # Store the output predictions in conveniently-accessible
        # attributes in the NAG, for easy downstream use of the saved
        # object
        if not output.multi_stage:
            logits = output.logits
            pred = torch.argmax(logits, dim=1)

            # Store level-1 predictions and logits
            batch[1].semantic_pred = pred
            batch[1].logits = logits

            # Store level-0 (voxel-wise) predictions and logits
            if getattr(output, 'point_logits', None) is not None:
                batch[0].logits = output.point_logits
                batch[0].semantic_pred = torch.argmax(output.point_logits, dim=1)
            else:
                batch[0].semantic_pred = pred[batch[0].super_index]
                batch[0].logits = logits[batch[0].super_index]

        else:
            for i, _logits in enumerate(output.logits):
                logits = _logits
                pred = torch.argmax(logits, dim=1)

                # Store level-1 predictions and logits
                batch[i + 1].semantic_pred = pred
                batch[i + 1].logits = logits

                # Store level-0 (voxel-wise) predictions and logits
                if i > 0:
                    continue
                if getattr(output, 'point_logits', None) is not None:
                    batch[0].logits = output.point_logits
                    batch[0].semantic_pred = torch.argmax(output.point_logits, dim=1)
                else:
                    batch[0].semantic_pred = pred[batch[0].super_index]
                    batch[0].logits = logits[batch[0].super_index]

        if hasattr(output, 'bsr_output') and output.bsr_output is not None:
            self._attach_bsr_tracking_metadata(batch, output.bsr_output)

        # Detach the batch object and move it to CPU before saving
        batch = batch.detach().cpu()

        # Prepare the folder
        try:
            if self.trainer is None:
                stage = 'unknown_stage'
            elif self.trainer.training:
                stage = 'train'
            elif self.trainer.validating:
                stage = 'val'
            elif self.trainer.testing:
                stage = 'test'
            elif self.trainer.predicting:
                stage = 'predict'
            else:
                stage = 'unknown_stage'
        except:
            stage = 'unknown_stage'
        if folder is None:
            if self.logger and self.logger.save_dir:
                folder = self.logger.save_dir
            else:
                folder = ''
        folder = osp.join(folder, 'predictions', stage, str(self.current_epoch))
        if not osp.isdir(folder):
            os.makedirs(folder, exist_ok=True)

        # Save to disk
        path = osp.join(folder, f"batch_{batch_idx}.h5")
        batch.save(path)
        log.info(f'Stored predictions at: "{path}"')

        # TODO: log plotly plot to wandb
        if isinstance(self.logger, WandbLogger):
            pass

    def configure_optimizers(self) -> Dict:
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#configure-optimizers
        """
        # Differential learning rate for transformer blocks
        t_names = ['transformer_blocks', 'down_pool_block']
        lr = self.hparams.optimizer.keywords['lr']
        t_lr = lr * self.hparams.transformer_lr_scale
        param_dicts = [
            {
                "params": [
                    p
                    for n, p in self.named_parameters()
                    if all([t not in n for t in t_names]) and p.requires_grad]},
            {
                "params": [
                    p
                    for n, p in self.named_parameters()
                    if any([t in n for t in t_names]) and p.requires_grad],
                "lr": t_lr}]
        optimizer = self.hparams.optimizer(params=param_dicts)

        # Return the optimizer if no scheduler in the config
        if self.hparams.scheduler is None:
            return {"optimizer": optimizer}

        # Build the scheduler, with special attention for plateau-like
        # schedulers, which
        scheduler = self.hparams.scheduler(optimizer=optimizer)
        reduce_on_plateau = isinstance(scheduler, ON_PLATEAU_SCHEDULERS)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val/loss",
                "interval": "epoch",
                "frequency": 1,
                "reduce_on_plateau": reduce_on_plateau}}

    def load_state_dict(
            self,
            state_dict: Dict,
            strict: bool = True
    ) -> None:
        """Basic `load_state_dict` from `torch.nn.Module` with a bit of
        acrobatics due to `criterion.weight`.

        This attribute, when present in the `state_dict`, causes
        `load_state_dict` to crash. More precisely, `criterion.weight`
        is holding the per-class weights for classification losses.
        """
        # Special treatment `criterion.weight`
        class_weight_bckp = self.criterion.weight
        self.criterion.weight = None

        # Recover the class weights from any `criterion.weight' or
        # 'criterion.*.weight' key and remove those keys from the
        # state_dict
        keys = []
        for key in state_dict.keys():
            if key.startswith('criterion.') and key.endswith('.weight'):
                keys.append(key)
        class_weight = state_dict[keys[0]] if len(keys) > 0 else None
        for key in keys:
            state_dict.pop(key)

        # Load the state_dict
        super().load_state_dict(state_dict, strict=strict)

        # If need be, assign the class weights to the criterion
        self.criterion.weight = class_weight if class_weight is not None \
            else class_weight_bckp

    def _load_from_checkpoint(
            self,
            checkpoint_path: str,
            **kwargs
    ) -> 'SemanticSegmentationModule':
        """Simpler version of `LightningModule.load_from_checkpoint()`
        for easier use: no need to explicitly pass `model.net`,
        `model.criterion`, etc.
        """
        return self.__class__.load_from_checkpoint(
            checkpoint_path,
            net=self.net,
            criterion=self.criterion,
            **kwargs)

    def on_save_checkpoint(self, checkpoint: dict) -> None:
        """
        Save metadata (version and commit hash) in the checkpoint.
        """
        # Add metadata to the checkpoint
        if 'metadata' not in checkpoint:
            checkpoint['metadata'] = {}
        
        # Update the checkpoint metadata with the version  and commit hash
        checkpoint['metadata']['__version__'] = self.net.version_holder.value
        checkpoint['metadata']['commit_hash'] = self.net.version_holder.commit_hash

    def on_load_checkpoint(self, checkpoint: dict) -> None:
        """
        Called when loading a checkpoint.
        Verifies version compatibility and logs information.
        """
        if 'metadata' not in checkpoint:
            checkpoint['metadata'] = {}
        
        # Affect default version if not found
        if '__version__' not in checkpoint['metadata']:
            version = '2.1.0'
            log.warning(
                "⚠️ No `__version__` found in checkpoint metadata."
                "\nThis means the checkpoint was saved with a version of the "
                "code prior to 3.0.0."
                f"\nSetting the version {version}, so that the official "
                f"weights of SPT and SPC are compatible."
                "\nIf you have weights from version 2.2.0, please use the "
                "migration script to set the version to 3.0.0."
                "\n(see: "
                "src/utils/backwards_compatibility/add_version_to_checkpoint.py"
                " and CHANGELOG.md for more details)")
            checkpoint['metadata']['__version__'] = version

        # Affect default commit hash if not found
        if 'commit_hash' not in checkpoint['metadata']:
            commit_hash = 'unknown'
            log.warning(
                "⚠️ No `commit_hash` found in checkpoint metadata."
                "\nThis means the checkpoint was saved with a version of the "
                "code prior to 3.0.0."
                f"\nSetting the commit hash to {commit_hash}.")
            checkpoint['metadata']['commit_hash'] = commit_hash
        
        # Update network version from checkpoint
        self.net.version = checkpoint['metadata']['__version__']
        self.net.version_holder.commit_hash = checkpoint['metadata']['commit_hash']

    @staticmethod
    def sanitize_step_output(out_dict: Dict) -> Dict:
        """Helper to be used for cleaning up the `_step` functions.
        Lightning expects those to return the loss (on GPU, with the
        computation graph intact for the backward step. Any other
        element passed in this dict will be detached and moved to CPU
        here. This avoids memory leak.
        """
        return {
            k: v if ((k == "loss") or (not isinstance(v, torch.Tensor)))
            else v.detach().cpu()
            for k, v in out_dict.items()}


class PartitionAndSemanticModule(SemanticSegmentationModule):
    """A LightningModule for semantic segmentation with two training stages.

    This module extends SemanticSegmentationModule to support two distinct training stages:
        1. First stage: train the model to partition the point cloud (into superpoints)
        2. Second stage: train the model to assign a semantic label to each superpoint.
            The preprocessing of this second stage partitions the point cloud based
            on the point features optimized during the first stage.

    The phase can be controlled by the boolean parameter `training_partition_stage`.

    :param training_partition_stage: bool
        If True, the model learns point features to build a good partition.
            The lightweight CNN computes point embeddings and is
            optimized for detecting semantic transitions.
            See the class PartitionCriterion for more details on
            the loss function.
        If False, the model is in the semantic segmentation stage.
            See behavior of `SemanticSegmentationModule`.
    :param partition : torch.nn.Module
        Module that computes a hierarchical partition.
        It takes a `Data` object (having notably the attributes `x`, `edge_index`) and
        returns a `NAG` object storing the partition.
        It is typically an instance of `src.transforms.partition.GreedyContourPriorPartition`.
    :param partition_criterion: torch.nn.Module
        It should be an instance of `src.loss.partition_criterion.PartitionCriterion`.
    :param partition_during_training: bool
        If True, the partition is computed during the training step.
        This is useful to get the partition metrics on the training set,
        but it significantly slows down the training procedure.
        Note: the partition is always computed during the validation step.
    """

    def __init__(
            self,
            net: torch.nn.Module,
            criterion: 'torch.nn._Loss',
            optimizer: torch.optim.Optimizer,
            scheduler: Any,
            num_classes: int,
            class_names: List[str] = None,
            sampling_loss: bool = False,
            loss_type: str = 'ce_kl',
            weighted_loss: bool = True,
            init_linear: str = None,
            init_rpe: str = None,
            transformer_lr_scale: float = 1,
            multi_stage_loss_lambdas: List[float] = None,
            gc_every_n_steps: int = 0,
            track_val_every_n_epoch: int = 1,
            track_val_idx: int = None,
            track_test_idx: int = None,

            training_partition_stage: bool = True,

            partition: torch.nn.Module = None,
            partition_criterion: torch.nn.Module = None,
            partition_during_training: bool = False,
            **kwargs):
        super().__init__(
            net=net,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            num_classes=num_classes,
            class_names=class_names,
            sampling_loss=sampling_loss,
            loss_type=loss_type,
            weighted_loss=weighted_loss,
            init_linear=init_linear,
            init_rpe=init_rpe,
            transformer_lr_scale=transformer_lr_scale,
            multi_stage_loss_lambdas=multi_stage_loss_lambdas,
            gc_every_n_steps=gc_every_n_steps,
            track_val_every_n_epoch=track_val_every_n_epoch,
            track_val_idx=track_val_idx,
            track_test_idx=track_test_idx,
            **kwargs)

        # Boolean flag to control the partition and classify phase
        self.training_partition_stage = training_partition_stage

        if self.training_partition_stage:
            del self.head

        # Module performing the partition
        self.partition = partition
        self.partition_criterion = partition_criterion
        self.partition_during_training = partition_during_training

        # Partition specific metrics
        self.train_partition_loss = MeanMetric()
        self.val_partition_loss = MeanMetric()
        self.test_partition_loss = MeanMetric()

        self.partition_train_cm = ConfusionMatrix(num_classes)
        self.partition_val_cm = ConfusionMatrix(num_classes)
        self.partition_test_cm = ConfusionMatrix(num_classes)

        self.val_partition_omiou_best = MaxMetric()
        self.val_partition_ooa_best = MaxMetric()
        self.val_partition_omacc_best = MaxMetric()

    def forward(
            self,
            sample: Union[NAG, Data],
    ) -> Union[SemanticSegmentationOutput, PartitionOutput]:
        # `sample` is a `Data` if the training_partition_stage is True
        # `sample` is a `NAG` otherwise

        if self.training_partition_stage:
            sample.add_keys_to(
                keys=self.net.point_hf,
                to='x',
                delete_after=not self.net.store_features)

            x, diameter = self.net.forward_first_stage(
                sample,
               first_stage=self.net.first_stage,
               use_node_hf=self.net.use_node_hf,
               norm_mode=self.net.norm_mode, )

            # Store the features on the level to be partitioned
            sample.x = x

            if (not self.training) or (self.partition_during_training):
                # self.partition is a Transform from data to NAG
                nag = self.partition(sample)
            else:
                nag = None

            # If the training procedure of the partition does not need
            # to compute the partition, `PartitionOutput` won't actually
            # hold a partition during training (unless
            # `partition_during_training` is set to True). Therefore,
            # the hard partition is given during validation epochs so
            # that the partition metrics are computed
            return PartitionOutput(
                y=sample.y,
                x=sample.x,
                edge_index=sample.edge_index,
                partition=nag[1] if nag is not None else None)

        else:
            return super().forward(sample)

    def model_step(
            self,
            batch: NAG
    ) -> Tuple[torch.Tensor, SemanticSegmentationOutput]:
        """Model step that changes based on current phase"""

        if self.training_partition_stage:

            partition_output = self.forward(batch)

            # If there are targets, compute the loss to train the
            # partition
            if partition_output.has_target:
                loss, partition_output = self.partition_criterion(
                    partition_output)
                return loss, partition_output
            else:
                return None, partition_output

        else:
            return super().model_step(batch)

    def train_step_update_metrics(self, loss, output) -> None:
        if not self.training_partition_stage:
            return super().train_step_update_metrics(loss, output)
        else:
            self.train_partition_loss(loss.detach())

            if self.partition_during_training:
                y_oracle = output.y_superpoint[:, :self.num_classes].argmax(dim=1)
                self.partition_train_cm(
                    pred=y_oracle,
                    target=output.y_superpoint)

    def train_step_log_metrics(self) -> None:
        if not self.training_partition_stage:
            return super().train_step_log_metrics()
        else:
            batch_size = self._batch_size_for_logging()
            self.log(
                "train/partition_loss",
                self.train_partition_loss,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                batch_size=batch_size)

    def on_train_epoch_end(self) -> None:
        if not self.training_partition_stage:
            return super().on_train_epoch_end()
        else:

            if self.partition_during_training:
                self._on_train_epoch_end(
                    cm=self.partition_train_cm,
                    metric_category='superpoints_purity')

    def validation_step_update_metrics(self, loss, output) -> None:

        if not self.training_partition_stage:
            return super().validation_step_update_metrics(loss, output)

        else:
            self.val_partition_loss(loss.detach())

            y_oracle = output.y_superpoint[:, :self.num_classes].argmax(dim=1)
            self.partition_val_cm(
                pred=y_oracle,
                target=output.y_superpoint)

            self.val_n_sp(output.partition.num_points)
            self.val_n_p(output.x.shape[0])

    def validation_step_log_metrics(self) -> None:
        """Log validation metrics after a single step with the content
        of the output object.
        """
        if not self.training_partition_stage:
            return super().validation_step_log_metrics()

        else:
            batch_size = self._batch_size_for_logging()
            self.log(
                "val/partition_loss",
                self.val_partition_loss,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                batch_size=batch_size)

            self.log(
                "val/n_sp",
                self.val_n_sp,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                batch_size=batch_size)

    def on_validation_epoch_end(self) -> None:
        if not self.training_partition_stage:
            return super().on_validation_epoch_end()
        else:
            self._on_eval_epoch_end(
                stage='val',
                cm=self.partition_val_cm,
                metric_category='superpoints_purity',
                miou_best=self.val_partition_omiou_best,
                oa_best=self.val_partition_ooa_best,
                macc_best=self.val_partition_omacc_best,
            )
            # Log ratio at the end of epoch when all metrics are
            # accumulated
            n_p = self.val_n_p.compute()
            n_sp = self.val_n_sp.compute()
            if n_sp > 0:
                self.log(
                    "val/points_per_superpoint",
                    n_p / n_sp,
                    prog_bar=True)

            self.val_n_p.reset()
            self.val_n_sp.reset()

    def test_step_update_metrics(self, loss, output) -> None:
        if not self.training_partition_stage:
            return super().test_step_update_metrics(loss, output)
        else:
            if not self.test_has_target:
                return
            self.test_partition_loss(loss.detach())

            y_oracle = output.y_superpoint[:, :self.num_classes].argmax(dim=1)
            self.partition_test_cm(
                pred=y_oracle,
                target=output.y_superpoint)
            
            self.test_n_sp(output.partition.num_points)
            self.test_n_p(output.x.shape[0])

    def test_step_log_metrics(self) -> None:
        if not self.training_partition_stage:
            return super().test_step_log_metrics()
        else:
            # If the test set misses targets, we keep track of it, to
            # skip metrics computation on the test set
            if not self.test_has_target:
                return

            batch_size = self._batch_size_for_logging()
            self.log(
                "test/partition_loss",
                self.test_partition_loss,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                batch_size=batch_size)

            self.log(
                "test/n_sp",
                self.test_n_sp,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                batch_size=batch_size)

    def on_test_epoch_end(self) -> None:
        if not self.training_partition_stage:
            return super().on_test_epoch_end()
        else:
            self._on_eval_epoch_end(
                stage='test',
                cm=self.partition_test_cm,
                metric_category='superpoints_purity')

            # Log ratio at the end of epoch when all metrics are
            # accumulated
            n_p = self.test_n_p.compute()
            n_sp = self.test_n_sp.compute()
            if n_sp > 0:
                self.log(
                    "test/points_per_superpoint",
                    n_p / n_sp,
                    prog_bar=True)

            self.test_n_p.reset()
            self.test_n_sp.reset()

    def _load_from_checkpoint(
            self,
            checkpoint_path: str,
            **kwargs
    ) -> 'SemanticSegmentationModule':
        """Simpler version of `LightningModule.load_from_checkpoint()`
        for easier use: no need to explicitly pass `model.net`,
        `model.criterion`, `model.partition`, etc.
        """
        return self.__class__.load_from_checkpoint(
            checkpoint_path, 
            net=self.net, 
            criterion=self.criterion, 
            partition=self.partition,
            **kwargs)


if __name__ == "__main__":
    import hydra
    import omegaconf
    import pyrootutils

    root = str(pyrootutils.setup_root(__file__, pythonpath=True))
    cfg = omegaconf.OmegaConf.load(root + "/configs/model/semantic/spt-2.yaml")
    _ = hydra.utils.instantiate(cfg)
