from typing import Optional

import torch.distributed as dist

from . import adapters, config, ops  # noqa: F401


def setup(**kwargs) -> None:
    dist.init_process_group(backend=config.dist.backend, world_size=config.dist.world_size, **kwargs)


def cleanup() -> None:
    dist.destroy_process_group()


def get_world_size() -> int:
    return dist.get_world_size()


def get_rank() -> int:
    return dist.get_rank()


def is_master(rank: Optional[int] = None) -> bool:
    if rank is None:
        rank = get_rank()
    return rank == 0
