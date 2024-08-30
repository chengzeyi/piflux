import os  # noqa: C101
import sys

# import torch


def optional_bool_from_env(name, default=None):
    val = os.environ.get(name, None)
    if val is None:
        return default
    if val == "1":
        return True
    return False


_save_config_ignore = {
    # workaround: "Can't pickle <function ...>"
}

world_size = optional_bool_from_env("PIFLUX_WORLD_SIZE")


class dist:
    backend = "nccl"


try:
    from torch.utils._config_module import install_config_module
except ImportError:
    # torch<2.2.0
    from torch._dynamo.config_utils import install_config_module

# adds patch, save_config, etc
install_config_module(sys.modules[__name__])
