from typing import Optional

import torch
import torch.distributed as dist

from piflux import context


@torch.library.custom_op(
    "piflux::get_assigned_chunk",
    mutates_args=(),
    device_types=[None, "meta"],
)
def get_assigned_chunk(
    tensor: torch.Tensor,
    dim: int = 0,
    idx: Optional[int] = None,
) -> torch.Tensor:
    ctx = context.current_context
    assert ctx is not None

    if idx is None:
        idx = ctx.offset

    return torch.chunk(tensor, ctx.world_size, dim=dim)[idx]


@torch.library.custom_op(
    "piflux::cat_from_gather_or_cache",
    mutates_args=(),
    device_types=[None, "meta"],
)
def cat_from_gather_or_cache(
    tensor: torch.Tensor,
    *,
    dim: int = 0,
    name: Optional[str] = None,
) -> torch.Tensor:
    ctx = context.current_context
    assert ctx is not None

    if name is None:
        idx = ctx.counters["cat_from_gather_or_cache"]
        ctx.counters["cat_from_gather_or_cache"] += 1
        name = f"cat_from_gather_or_cache_{idx}"

    gathered_tensors = ctx.get_buffer_list(tensor, name=name)

    if tensor.device.type != "meta":
        if ctx.is_sync_step:
            dist.all_gather(gathered_tensors, tensor.contiguous())
            world_size = ctx.world_size
            master_offset = ctx.master_offset
            gathered_tensors = (
                gathered_tensors[world_size - master_offset :] + gathered_tensors[: world_size - master_offset]
            )
        else:
            offset = ctx.offset
            gathered_tensors[offset] = tensor

        ctx.set_buffer_list(name, gathered_tensors)

    output_shape = list(tensor.shape)
    output_shape[dim] *= len(gathered_tensors)
    output_tensor = torch.empty(output_shape, dtype=tensor.dtype, device=tensor.device)
    output_tensor = torch.cat(gathered_tensors, dim=dim, out=output_tensor)
    return output_tensor


@torch.library.custom_op(
    "piflux::cat_from_gather",
    mutates_args=(),
    device_types=[None, "meta"],
)
def cat_from_gather(
    tensor: torch.Tensor,
    *,
    dim: int = 0,
) -> torch.Tensor:
    ctx = context.current_context
    assert ctx is not None

    gathered_tensors = ctx.get_buffer_list(tensor)

    if tensor.device.type != "meta":
        dist.all_gather(gathered_tensors, tensor.contiguous())
        world_size = ctx.world_size
        master_offset = ctx.master_offset
        gathered_tensors = (
            gathered_tensors[world_size - master_offset :] + gathered_tensors[: world_size - master_offset]
        )

    output_shape = list(tensor.shape)
    output_shape[dim] *= len(gathered_tensors)
    output_tensor = torch.empty(output_shape, dtype=tensor.dtype, device=tensor.device)
    output_tensor = torch.cat(gathered_tensors, dim=dim, out=output_tensor)
    return output_tensor
