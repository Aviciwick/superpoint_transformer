import torch
import hydra
from src.utils.hydra import init_config

# 1. Baseline
cfg_base = init_config(config_name="train.yaml", overrides=["experiment=panoptic/kitti360"])
dm_base = hydra.utils.instantiate(cfg_base.datamodule)
print("Base point_load_keys:", dm_base._point_load_keys)

# 2. HSPT
cfg_hspt = init_config(config_name="train.yaml", overrides=["experiment=panoptic/kitti360", "model.hspt.enable=True"])
dm_hspt = hydra.utils.instantiate(cfg_hspt.datamodule)
print("HSPT point_load_keys:", dm_hspt._point_load_keys)

