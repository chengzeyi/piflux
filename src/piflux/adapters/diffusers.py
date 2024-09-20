try:
    import diffusers  # noqa: F401
except ImportError:
    raise ImportError("diffusers is not installed. Install it by `pip3 install diffusers`")

import functools
from typing import List, Optional

import torch

from diffusers import DiffusionPipeline, FluxTransformer2DModel

from piflux import context

from piflux.mode import DistributedAttentionMode

piflux_ops = torch.ops.piflux


def patch_transformer(transformer: FluxTransformer2DModel) -> None:
    assert isinstance(transformer, FluxTransformer2DModel)

    original_forward = transformer.forward

    @functools.wraps(original_forward.__func__)
    @torch.compiler.disable(recursive=False)
    def new_forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        *args,
        img_ids: torch.Tensor = None,
        txt_ids: torch.Tensor = None,
        controlnet_block_samples: Optional[List[torch.Tensor]] = None,
        controlnet_single_block_samples: Optional[List[torch.Tensor]] = None,
        **kwargs,
    ):
        hidden_states = piflux_ops.get_assigned_chunk(hidden_states, dim=-2)
        encoder_hidden_states = piflux_ops.get_assigned_chunk(encoder_hidden_states, dim=-2)
        img_ids = piflux_ops.get_assigned_chunk(img_ids, dim=-2)
        txt_ids = piflux_ops.get_assigned_chunk(txt_ids, dim=-2)
        if controlnet_block_samples is not None:
            controlnet_block_samples = [
                piflux_ops.get_assigned_chunk(sample, dim=-2) for sample in controlnet_block_samples
            ]
            kwargs["controlnet_block_samples"] = controlnet_block_samples
        if controlnet_single_block_samples is not None:
            controlnet_single_block_samples = [
                piflux_ops.get_assigned_chunk(sample, dim=-2) for sample in controlnet_single_block_samples
            ]
            kwargs["controlnet_single_block_samples"] = controlnet_single_block_samples

        with DistributedAttentionMode():
            output = original_forward(
                hidden_states,
                encoder_hidden_states,
                *args,
                img_ids=img_ids,
                txt_ids=txt_ids,
                **kwargs,
            )
        return_dict = not isinstance(output, tuple)
        sample = output[0]

        sample = piflux_ops.get_complete_tensor(sample, dim=-2)

        sample = piflux_ops.next_step(sample)

        if return_dict:
            return output.__class__(sample, *output[1:])
        return (sample, *output[1:])

    new_forward = new_forward.__get__(transformer)
    transformer.forward = new_forward


def patch_pipe(pipe: DiffusionPipeline) -> None:
    assert isinstance(pipe, DiffusionPipeline)
    patch_transformer(pipe.transformer)

    original_prepare_latents = pipe.prepare_latents

    @functools.wraps(pipe.prepare_latents.__func__)
    def new_prepare_latents(self, *args, **kwargs):
        latents, latent_image_ids = original_prepare_latents(*args, **kwargs)
        latents = piflux_ops.get_complete_tensor(latents, dim=0)
        latents = piflux_ops.get_assigned_chunk(latents, dim=0, idx=0)
        return latents, latent_image_ids

    pipe.prepare_latents = new_prepare_latents.__get__(pipe)

    original_call = pipe.__class__.__call__

    @functools.wraps(original_call)
    def new_call(self, *args, **kwargs):
        ctx = context.create_context()
        with context.patch_current_context(ctx):
            return original_call(self, *args, **kwargs)

    pipe.__class__.__call__ = new_call
