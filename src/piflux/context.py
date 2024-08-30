import contextlib
import dataclasses
from collections import defaultdict
from typing import DefaultDict, Dict, List, Optional, Tuple, Union

import torch

current_context = None


@dataclasses.dataclass
class ParallelContext:
    world_size: int
    rank: int

    buffers: Dict[str, torch.Tensor] = dataclasses.field(default_factory=dict)
    buffer_lists: Dict[str, List[torch.Tensor]] = dataclasses.field(default_factory=dict)

    counters: DefaultDict[str, int] = dataclasses.field(default_factory=lambda: defaultdict(int))

    step: int = 0

    def next_step(self) -> None:
        self.step += 1
        self.counters.clear()

    @property
    def offset(self) -> int:
        return (self.rank + self.step) % self.world_size

    @property
    def master_offset(self) -> int:
        return self.step % self.world_size


@contextlib.contextmanager
def patch_current_context(new_context: ParallelContext) -> None:
    global current_context
    old_context = current_context
    current_context = new_context
    try:
        yield
    finally:
        current_context = old_context


def get_buffer(
    name: str,
    shape_or_tensor: Union[Tuple[int], torch.Tensor],
    *,
    repeats: Optional[int] = None,
    dim=0,
    dtype: Optional[torch.dtype] = None,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    assert current_context is not None

    if repeats is None:
        repeats = current_context.world_size

    if isinstance(shape_or_tensor, torch.Tensor):
        shape = shape_or_tensor.shape
        dtype = shape_or_tensor.dtype
        device = shape_or_tensor.device

    assert dtype is not None
    assert device is not None

    buffer = current_context.buffers.get(name)
    if buffer is None or buffer.shape != shape or buffer.dtype != dtype or buffer.device != device:
        if repeats > 1:
            new_shape = list(shape)
            new_shape[dim] *= repeats
            buffer = torch.empty(new_shape, dtype=dtype, device=device)
        else:
            buffer = (
                torch.empty_like(shape_or_tensor)
                if isinstance(shape_or_tensor, torch.Tensor)
                else torch.empty(shape, dtype=dtype, device=device)
            )
        current_context.buffers[name] = buffer
    return buffer


def get_buffer_list(
    name: str,
    shape_or_tensor: Union[Tuple[int], torch.Tensor],
    *,
    num: Optional[int] = None,
    dtype: Optional[torch.dtype] = None,
    device: Optional[torch.device] = None,
) -> List[torch.Tensor]:
    assert current_context is not None

    if num is None:
        num = current_context.world_size

    if isinstance(shape_or_tensor, torch.Tensor):
        shape = shape_or_tensor.shape
        dtype = shape_or_tensor.dtype
        device = shape_or_tensor.device

    assert dtype is not None
    assert device is not None

    buffer_list = current_context.buffer_lists.get(name)
    if (
        buffer_list is None
        or len(buffer_list) != num
        or buffer_list[0].shape != shape
        or buffer_list[0].dtype != dtype
        or buffer_list[0].device != device
    ):
        buffer_list = [
            torch.empty_like(shape_or_tensor)
            if isinstance(shape_or_tensor, torch.Tensor)
            else torch.empty(shape, dtype=dtype, device=device)
            for _ in range(num)
        ]
        current_context.buffer_lists[name] = buffer_list
    return buffer_list
