import importlib

import pytest

import torch

if not torch.cuda.is_available():
    pytest.skip("CUDA is not available", allow_module_level=True)

if torch.cuda.device_count() < 2:
    pytest.skip("At least two CUDA devices are required", allow_module_level=True)

if not importlib.util.find_spec("diffusers"):
    pytest.skip("diffusers is not available", allow_module_level=True)

import dataclasses
import os
import pickle
import socket
from contextlib import closing
from datetime import timedelta
from typing import Any, Dict

import piflux
import torch.multiprocessing as mp

from diffusers import DiffusionPipeline
from piflux.utils.term_image import print_image


def find_free_port():
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(("", 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]


@dataclasses.dataclass
class SharedValue:
    input_kwargs: Dict[str, Any]
    debug_raise_exception: bool = dataclasses.field(default=False)


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


def worker(
    rank,
    barrier,
    array_size,
    shared_array,
    has_error,
    input_queue=None,
    output_queue=None,
):
    piflux.config.dist.world_size = 2

    def init_piflux():
        piflux.setup(rank=rank, timeout=timedelta(seconds=10))

    def cleanup_piflux():
        piflux.cleanup()

    init_piflux()

    with torch.cuda.device(rank):
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
            debug_raise_exception = value.debug_raise_exception
            output = None
            exception = None
            try:
                if debug_raise_exception and piflux.is_master(rank):
                    raise RuntimeError("Debug exception")
            except Exception as e:
                exception = e

            if exception is not None:
                print(f"Rank {rank} preparation failed with exception: {exception}")
                with has_error.get_lock():
                    has_error.value = 1

            barrier.wait()
            if bool(has_error.value):
                if output_queue is not None:
                    if exception is not None:
                        output_queue.put(exception)
                    else:
                        output_queue.put(RuntimeError("Exception occurred"))
                barrier.wait()
                if piflux.is_master(rank):
                    has_error.value = 0
                continue

            try:
                if debug_raise_exception and piflux.is_master(rank):
                    raise RuntimeError("Debug exception")
                output = call_pipe(pipe, **input_kwargs)
            except Exception as e:
                exception = e
            if exception is not None:
                print(f"Rank {rank} inference failed with exception: {exception}")
                with has_error.get_lock():
                    has_error.value = 1
            barrier.wait()
            if output_queue is not None:
                if output is not None:
                    output_queue.put(output)
                elif exception is not None:
                    output_queue.put(exception)
                else:
                    raise RuntimeError("No output or exception")
            if bool(has_error.value):
                print(f"Rank {rank} restarting")
                cleanup_piflux()
                init_piflux()
                barrier.wait()
                if piflux.is_master(rank):
                    has_error.value = 0


def init_process(
    rank,
    barrier,
    array_size,
    shared_array,
    has_error,
    input_queue,
    output_queue,
):
    try:
        if not piflux.is_master(rank):
            input_queue, output_queue = None, None
        worker(
            rank,
            barrier,
            array_size,
            shared_array,
            has_error,
            input_queue,
            output_queue,
        )
    finally:
        piflux.cleanup()


def call_once(
    processes, array_size, shared_array, input_queue, output_queue, input_kwargs=None, debug_raise_exception=False
):
    for rank, process in enumerate(processes):
        if not process.is_alive():
            raise RuntimeError(f"Process {rank} is not alive")

    input_kwargs = input_kwargs or {}
    data = pickle.dumps(SharedValue(input_kwargs, debug_raise_exception))
    data_size = len(data)
    array_size.value = data_size
    shared_array[:data_size] = data
    input_queue.put(True)
    output = output_queue.get(timeout=30)
    if isinstance(output, Exception):
        assert debug_raise_exception
        print(f"Exception: {output}")
    else:
        assert not debug_raise_exception
        for image in output.images:
            print_image(image, max_width=60)


def test_subprocess():
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(find_free_port())

    num_workers = 2
    mp.set_start_method("spawn")
    barrier = mp.Barrier(num_workers)
    array_size = mp.Value("i", 0)
    shared_array = mp.Array("c", 1 << 30, lock=False)
    has_error = mp.Value("i", 0)
    input_queue = mp.Queue()
    output_queue = mp.Queue()
    processes = []
    try:
        for rank in range(num_workers):
            process = mp.Process(
                target=init_process,
                args=(
                    rank,
                    barrier,
                    array_size,
                    shared_array,
                    has_error,
                    input_queue,
                    output_queue,
                ),
            )
            process.start()
            processes.append(process)

        input_kwargs = {
            "prompt": "A cat holding a sign that says hello world",
            "num_inference_steps": 28,
            "height": 1024,
            "width": 1024,
            "seed": 0,
        }

        call_once(processes, array_size, shared_array, input_queue, output_queue, input_kwargs=input_kwargs)
        call_once(processes, array_size, shared_array, input_queue, output_queue, debug_raise_exception=True)
        call_once(processes, array_size, shared_array, input_queue, output_queue, input_kwargs=input_kwargs)

        array_size.value = 0
        input_queue.put(None)
        for process in processes:
            process.join()
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
