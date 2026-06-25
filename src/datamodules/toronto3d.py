import logging

from src.datamodules.base import BaseDataModule
from src.datasets import MiniToronto3D, Toronto3D


log = logging.getLogger(__name__)


class Toronto3DDataModule(BaseDataModule):
    """LightningDataModule for the Toronto-3D placeholder dataset."""

    _DATASET_CLASS = Toronto3D
    _MINIDATASET_CLASS = MiniToronto3D
