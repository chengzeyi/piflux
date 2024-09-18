from typing import Optional

import torch
import torch.distributed as dist

from piflux import context


def next_step(
    tensor: torch.Tensor,
) -> torch.Tensor:
    ctx = context.current_context
    assert ctx is not None

    ctx.next_step()

    return tensor.clone()


def next_step_fake(
    tensor: torch.Tensor,
) -> torch.Tensor:
    return torch.empty_like(tensor)


torch.library.custom_op("piflux::next_step", mutates_args=(),)(
    next_step
).register_fake(next_step_fake)


def get_assigned_chunk(
    tensor: torch.Tensor,
    dim: int = 0,
    idx: Optional[int] = None,
) -> torch.Tensor:
    ctx = context.current_context
    assert ctx is not None

    if idx is None:
        idx = ctx.offset

    return tensor.chunk(ctx.world_size, dim=dim)[idx].clone()


torch.library.custom_op("piflux::get_assigned_chunk", mutates_args=(),)(
    get_assigned_chunk
).register_fake(get_assigned_chunk)


def get_complete_tensor(
    tensor: torch.Tensor,
    *,
    dim: int = 0,
    name: Optional[str] = None,
    enable_cache: bool = False,
) -> torch.Tensor:
    ctx = context.current_context
    assert ctx is not None

    if enable_cache:
        name = ctx.get_incremental_name(name)
    else:
        assert name is None

    world_size = ctx.world_size
    offset = ctx.offset
    master_offset = ctx.master_offset

    permute_dims = list(range(tensor.dim()))
    permute_dims[dim], permute_dims[0] = permute_dims[0], permute_dims[dim]
    tensor = tensor.permute(permute_dims)

    output_tensor = ctx.get_buffer(tensor, name=name, repeats=world_size, dim=0)

    if not enable_cache or ctx.is_sync_step:
        gathered_tensors = list(output_tensor.chunk(world_size, dim=0))
        gathered_tensors = (
            gathered_tensors[world_size - master_offset :] + gathered_tensors[: world_size - master_offset]
        )
        dist.all_gather(gathered_tensors, tensor.contiguous())
    else:
        gathered_tensor_shape = output_tensor.shape
        tmp_shape = list(output_tensor.shape)
        tmp_shape[0] //= world_size
        tmp_shape.insert(0, world_size)
        output_tensor = output_tensor.view(tmp_shape)
        output_tensor[offset].copy_(tensor)
        output_tensor = output_tensor.view(gathered_tensor_shape)

    ctx.set_buffer(name, output_tensor)

    output_tensor = output_tensor.permute(permute_dims)

    return output_tensor


def get_complete_tensor_fake(
    tensor: torch.Tensor,
    *,
    dim: int = 0,
    name: Optional[str] = None,
    enable_cache: bool = False,
) -> torch.Tensor:
    ctx = context.current_context
    assert ctx is not None

    output_shape = list(tensor.shape)
    output_shape[dim] *= ctx.world_size
    output_tensor = torch.empty(output_shape, dtype=tensor.dtype, device=tensor.device)

    permute_dims = list(range(tensor.dim()))
    permute_dims[dim], permute_dims[0] = permute_dims[0], permute_dims[dim]
    output_tensor = output_tensor.permute(permute_dims)
    output_tensor = output_tensor.contiguous()
    output_tensor = output_tensor.permute(permute_dims)

    return output_tensor


torch.library.custom_op("piflux::get_complete_tensor", mutates_args=(),)(
    get_complete_tensor
).register_fake(get_complete_tensor_fake)
