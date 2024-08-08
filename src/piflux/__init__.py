from diffusers import FluxPipeline, FluxTransformer2DModel
from diffusers.models.transformers.transformer_flux import FluxSingleTransformerBlock, FluxTransformerBlock

from . import config, patch  # noqa: F401


def patch_pipe(pipe: FluxPipeline) -> None:
    assert isinstance(pipe, FluxPipeline)
    patch_transformer(pipe.transformer)


def patch_transformer(transformer: FluxTransformer2DModel) -> None:
    assert isinstance(transformer, FluxTransformer2DModel)
    for name, module in transformer.named_modules():
        if isinstance(module, FluxTransformerBlock):
            module.__class__ = patch.make_parallel_flux_transformer_block(module.__class__)
        elif isinstance(module, FluxSingleTransformerBlock):
            pass
