import contextlib
import dataclasses
from collections import defaultdict
from typing import DefaultDict, Dict, List, Optional, Tuple, Union

import torch
import torch.distributed as dist

import piflux

current_context = None


@dataclasses.dataclass
class ParallelContext:
    world_size: int
    rank: int

    buffers: Dict[str, torch.Tensor] = dataclasses.field(default_factory=dict)
    buffer_lists: Dict[str, List[torch.Tensor]] = dataclasses.field(default_factory=dict)

    counters: DefaultDict[str, int] = dataclasses.field(default_factory=lambda: defaultdict(int))

    step: int = 0
    sync_steps: int = 1

    def next_step(self) -> None:
        self.step += 1
        total_count = 0
        for name, count in self.counters.items():
            total_count += count
        self.counters.clear()

    @property
    def offset(self) -> int:
        if self.sync_steps <= 0:
            return self.rank
        return (self.rank + max(0, self.step - self.sync_steps)) % self.world_size

    @property
    def master_offset(self) -> int:
        if self.sync_steps <= 0:
            return 0
        return max(0, self.step - self.sync_steps) % self.world_size

    @property
    def is_sync_step(self) -> bool:
        return self.sync_steps > 0 and self.step < self.sync_steps

    def get_incremental_name(self, name: Optional[str] = None) -> str:
        if name is None:
            name = "default"
        idx = self.counters[name]
        self.counters[name] += 1
        return f"{name}_{idx}"

    def get_buffer(
        self,
        shape_or_tensor: Union[Tuple[int], torch.Tensor],
        *,
        name: Optional[str] = None,
        repeats: int = 1,
        dim: int = 0,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        if repeats is None:
            repeats = self.world_size

        if isinstance(shape_or_tensor, torch.Tensor):
            shape = shape_or_tensor.shape
            dtype = shape_or_tensor.dtype
            device = shape_or_tensor.device

        assert dtype is not None
        assert device is not None

        shape = list(shape)
        if repeats > 1:
            shape[dim] *= repeats

        buffer = self.buffers.get(name)
        if buffer is None or list(buffer.shape) != shape or buffer.dtype != dtype or buffer.device != device:
            buffer = torch.empty(shape, dtype=dtype, device=device)
            if name is not None:
                self.buffers[name] = buffer
        return buffer

    def set_buffer(self, name: str, tensor: torch.Tensor) -> None:
        self.buffers[name] = tensor

    def get_buffer_list(
        self,
        shape_or_tensor: Union[Tuple[int], torch.Tensor],
        *,
        name: Optional[str] = None,
        num: Optional[int] = None,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
    ) -> List[torch.Tensor]:
        if num is None:
            num = self.world_size

        if isinstance(shape_or_tensor, torch.Tensor):
            shape = shape_or_tensor.shape
            dtype = shape_or_tensor.dtype
            device = shape_or_tensor.device

        assert dtype is not None
        assert device is not None

        shape = list(shape)

        buffer_list = self.buffer_lists.get(name)
        if (
            buffer_list is None
            or len(buffer_list) != num
            or list(buffer_list[0].shape) != shape
            or buffer_list[0].dtype != dtype
            or buffer_list[0].device != device
        ):
            buffer_list = [torch.empty(shape, dtype=dtype, device=device) for _ in range(num)]
            if name is not None:
                self.buffer_lists[name] = buffer_list
        return buffer_list

    def set_buffer_list(self, name: str, tensors: List[torch.Tensor]) -> None:
        self.buffer_lists[name] = tensors


def create_context() -> ParallelContext:
    world_size = dist.get_world_size()
    rank = dist.get_rank()

    return ParallelContext(world_size=world_size, rank=rank, sync_steps=piflux.config.sync_steps)


@contextlib.contextmanager
def patch_current_context(new_context: ParallelContext) -> None:
    global current_context
    old_context = current_context
    current_context = new_context
    try:
        yield
    finally:
        current_context = old_context
