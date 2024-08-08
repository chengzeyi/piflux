# PIFLUX

Parallel inference for `black-forest-labs`' [`FLUX`](https://huggingface.co/black-forest-labs/FLUX.1-schnell) model.

## Description

**PIFLUX** is a parallel inference optimization library for `black-forest-labs`' [`FLUX`](https://huggingface.co/black-forest-labs/FLUX.1-schnell) model. It works with `torch.distributed` to utilize more than 1 NVIDIA GPU to parallelize the inference to reduce the time needed for generating one image.

This library is only for demonstrative perpose and is not intended for production use. Whether it can be combined with other optimization techniques like `torch.compile` is not tested.
