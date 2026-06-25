import logging

from src.datamodules.base import BaseDataModule
from src.datasets import MiniSensatUrban, SensatUrban


log = logging.getLogger(__name__)


class SensatUrbanDataModule(BaseDataModule):
    """LightningDataModule for SensatUrban semantic segmentation."""

    _DATASET_CLASS = SensatUrban
    _MINIDATASET_CLASS = MiniSensatUrban
