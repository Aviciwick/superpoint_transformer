import os
from hydra import initialize, compose
from hydra.core.global_hydra import GlobalHydra


__all__ = ['init_config']


def init_config(config_name='train.yaml', overrides=[]):
    # Registering the "eval" resolver allows for advanced config
    # interpolation with arithmetic operations:
    # https://omegaconf.readthedocs.io/en/2.3_branch/how_to_guides.html
    from omegaconf import OmegaConf
    if not OmegaConf.has_resolver('eval'):
        OmegaConf.register_new_resolver('eval', eval)

    GlobalHydra.instance().clear()
    # Ensure CWD is this utils directory so relative config_path works
    cwd_bckp = os.getcwd()
    os.chdir(os.path.dirname(__file__))
    try:
        with initialize(config_path="../../configs"):
            cfg = compose(config_name=config_name, overrides=overrides)
    finally:
        os.chdir(cwd_bckp)
    return cfg
