import importlib

import pytest

import torch

if not torch.cuda.is_available():
    pytest.skip("CUDA is not available", allow_module_level=True)

if torch.cuda.device_count() < 2:
    pytest.skip("At least two CUDA devices are required", allow_module_level=True)

if not importlib.util.find_spec("diffusers"):
    pytest.skip("diffusers is not available", allow_module_level=True)

import os
import pickle
import socket
from contextlib import closing
from typing import Dict, Any
import dataclasses
import torch.multiprocessing as mp

from diffusers import DiffusionPipeline
import piflux
from piflux.utils.term_image import print_image


def find_free_port():
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(('', 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]


@dataclasses.dataclass
class SharedValue:
    input_kwargs: Dict[str, Any]


def load_pipe():
    pipe = DiffusionPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-dev",
        torch_dtype=torch.bfloat16,
    ).to("cuda")
    return pipe


def call_pipe(
    pipe,
    prompt="A cat holding a sign that says hello world",
    num_inference_steps=28,
    height=1024,
    width=1024,
    seed=0,
    **kwargs,
):
    generator = torch.Generator(device="cuda").manual_seed(seed)
    output = pipe(
        prompt=prompt,
        num_inference_steps=num_inference_steps,
        height=height,
        width=width,
        generator=generator,
        **kwargs,
    )
    return output


def worker(barrier, array_size, shared_array, input_queue=None, output_queue=None):
    with torch.cuda.device(piflux.get_rank()):
        pipe = load_pipe()
        piflux.adapters.diffusers.patch_pipe(pipe)
        while True:
            if input_queue is None:
                barrier.wait()
            else:
                data = input_queue.get()
                barrier.wait()
                if data is None:
                    break
            data_size = array_size.value
            if data_size == 0:
                break
            value_bytes = shared_array[:data_size]
            value = pickle.loads(value_bytes)
            input_kwargs = value.input_kwargs
            output = call_pipe(pipe, **input_kwargs)
            if output_queue is not None:
                output_queue.put(output)


def init_process(rank, barrier, array_size, shared_array, input_queue, output_queue):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(find_free_port())
    piflux.config.dist.world_size = 2
    piflux.setup(rank=rank)
    try:
        if not piflux.is_master():
            input_queue, output_queue = None, None
        worker(barrier, array_size, shared_array, input_queue, output_queue)
    finally:
        piflux.cleanup()


def test_subprocess():
    num_workers = 2
    mp.set_start_method("spawn")
    barrier = mp.Barrier(num_workers)
    array_size = mp.Value("i", 0)
    shared_array = mp.Array("c", 1 << 30, lock=False)
    input_queue = mp.Queue()
    output_queue = mp.Queue()
    processes = []
    try:
        for rank in range(num_workers):
            process = mp.Process(target=init_process, args=(rank, barrier, array_size, shared_array, input_queue, output_queue))
            process.start()
            processes.append(process)

        data = {
            "prompt": "A cat holding a sign that says hello world",
            "num_inference_steps": 28,
            "height": 1024,
            "width": 1024,
            "seed": 0,
        }
        data = pickle.dumps(SharedValue(data))
        data_size = len(data)
        assert data_size <= len(shared_array)
        array_size.value = data_size
        shared_array[:data_size] = data
        input_queue.put(True)
        output = output_queue.get()
        for image in output.images:
            print_image(image, max_width=60)

        array_size.value = 0
        input_queue.put(None)
        for process in processes:
            process.join()
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
