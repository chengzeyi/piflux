import os  # noqa: C101
import sys

# import torch


_save_config_ignore = {
    # workaround: "Can't pickle <function ...>"
}

sync_steps = int(os.environ.get("PIFLUX_SYNC_STEPS", "1"))


try:
    from torch.utils._config_module import install_config_module
except ImportError:
    # torch<2.2.0
    from torch._dynamo.config_utils import install_config_module

# adds patch, save_config, etc
install_config_module(sys.modules[__name__])
