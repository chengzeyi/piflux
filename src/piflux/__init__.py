from typing import Optional

import torch.distributed as dist

from . import adapters, config, ops  # noqa: F401

try:
    from ._version import version as __version__, version_tuple
except ImportError:
    __version__ = "unknown version"
    version_tuple = (0, 0, "unknown version")
