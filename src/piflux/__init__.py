import functools

import torch
import torch.distributed as dist
from diffusers import FluxPipeline, FluxTransformer2DModel

from . import config, context
from .mode import DistributedAttentionMode


def setup() -> None:
    world_size = config.world_size
    if world_size is None:
        world_size = torch.cuda.device_count()
    assert world_size > 0

    dist.init_process_group(backend=config.dist.backend, world_size=world_size)


def cleanup() -> None:
    dist.destroy_process_group()


def get_world_size() -> int:
    return dist.get_world_size()


def get_rank() -> int:
    return dist.get_rank()


def create_context() -> context.ParallelContext:
    world_size = get_world_size()
    rank = get_rank()

    return context.ParallelContext(world_size=world_size, rank=rank)


def patch_pipe(pipe: FluxPipeline) -> None:
    assert isinstance(pipe, FluxPipeline)
    patch_transformer(pipe.transformer)

    original_prepare_latents = pipe.prepare_latents

    @functools.wraps(pipe.prepare_latents.__func__)
    def new_prepare_latents(self, *args, **kwargs):
        ctx = context.current_context
        assert ctx is not None

        latents, latent_image_ids = original_prepare_latents(*args, **kwargs)
        latents = latents.contiguous()
        gathered_latents = context.get_buffer_list("pipe_prepare_latents_gathered_latents", latents)
        dist.all_gather(gathered_latents, latents)
        latents = gathered_latents[0]
        return latents, latent_image_ids

    pipe.prepare_latents = new_prepare_latents.__get__(pipe)

    original_call = pipe.__class__.__call__

    @functools.wraps(original_call)
    def new_call(self, *args, **kwargs):
        ctx = create_context()
        with context.patch_current_context(ctx):
            return original_call(self, *args, **kwargs)

    pipe.__class__.__call__ = new_call


def patch_transformer(transformer: FluxTransformer2DModel) -> None:
    assert isinstance(transformer, FluxTransformer2DModel)

    original_forward = transformer.forward

    @functools.wraps(original_forward.__func__)
    def new_forward(self, hidden_states: torch.Tensor, *args, img_ids: torch.Tensor = None, **kwargs):
        ctx = context.current_context
        assert ctx is not None
        rank = ctx.rank

        hidden_states = hidden_states.chunk(ctx.world_size, dim=1)[rank]
        img_ids = img_ids.chunk(ctx.world_size, dim=0)[rank]

        with DistributedAttentionMode():
            output = original_forward(hidden_states, *args, img_ids=img_ids, **kwargs)
        return_dict = not isinstance(output, tuple)
        sample = output[0]
        sample = sample.contiguous()

        gathered_sample = context.get_buffer("transformer_forward_gathered_sample", sample, dim=1)
        gathered_samples = context.get_buffer_list("transformer_forward_gathered_samples", sample)

        dist.all_gather(gathered_samples, sample)
        torch.cat(gathered_samples, dim=1, out=gathered_sample)

        ctx.next_step()

        if return_dict:
            return (gathered_sample, *output[1:])
        return output.__class__(gathered_sample, *output[1:])

    new_forward = new_forward.__get__(transformer)
    transformer.forward = new_forward
