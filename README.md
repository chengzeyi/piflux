# PIFLUX

Parallel inference for `black-forest-labs`' [`FLUX`](https://huggingface.co/black-forest-labs/FLUX.1-dev) model.

## Description

**PIFLUX** is a parallel inference optimization library for `black-forest-labs`' [`FLUX`](https://huggingface.co/black-forest-labs/FLUX.1-dev) model. It works with `torch.distributed` to utilize more than 1 NVIDIA GPU to parallelize the inference. So that the time needed for generating one image can be reduced more.

This library is only for demonstrative perpose and is not intended for production use. Whether it can be combined with other optimization techniques like `torch.compile` is not tested.

## Installation

```bash
git clone https://github.com/chengzeyi/piflux.git
cd xelerate
git submodule update --init --recursive

pip3 install packaging wheel 'setuptools>=64,<70' 'setuptools_scm>=8'

pip3 install -e '.[dev]'

# Code formatting and linting
pip3 install pre-commit
pre-commit install
pre-commit run --all-files
```

## Usage

### Run the example

```bash
N_GPUS=2
torchrun --nproc_per_node=$N_GPUS examples/run_flux.py --print-output
```

## Thanks

- [xDiT](https://github.com/xdit-project/xDiT) for the [PipeFusion](https://github.com/xdit-project/xDiT/blob/main/docs/methods/pipefusion.md) parallel method.
