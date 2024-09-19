import torch.distributed as dist

from . import adapters, config, ops  # noqa: F401


def setup() -> None:
    world_size = config.world_size

    dist.init_process_group(backend=config.dist.backend, world_size=world_size)


def cleanup() -> None:
    dist.destroy_process_group()


def get_world_size() -> int:
    return dist.get_world_size()


def get_rank() -> int:
    return dist.get_rank()


def is_master() -> bool:
    return get_rank() == 0
