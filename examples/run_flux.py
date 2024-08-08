MODEL = "black-forest-labs/FLUX.1-schnell"
VARIANT = None
DTYPE = "bfloat16"
DEVICE = "cuda"
PIPELINE_CLASS = "FluxPipeline"
CUSTOM_PIPELINE = None
SCHEDULER = None
LORA = None
CONTROLNET = None
STEPS = 4
PROMPT = "A cat holding a sign that says hello world"
NEGATIVE_PROMPT = None
SEED = None
WARMUPS = 1
BATCH = 1
HEIGHT = None
WIDTH = None
INPUT_IMAGE = None
CONTROL_IMAGE = None
OUTPUT_IMAGE = None
EXTRA_CALL_KWARGS = None

WORLD_SIZE = None

import argparse
import importlib
import inspect
import itertools
import json
import time

import diffusers

import torch
from diffusers.utils import load_image
from PIL import Image, ImageDraw


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=MODEL)
    parser.add_argument("--variant", type=str, default=VARIANT)
    parser.add_argument("--dtype", type=str, default=DTYPE)
    parser.add_argument("--device", type=str, default=DEVICE)
    parser.add_argument("--pipeline-class", type=str, default=PIPELINE_CLASS)
    parser.add_argument("--custom-pipeline", type=str, default=CUSTOM_PIPELINE)
    parser.add_argument("--scheduler", type=str, default=SCHEDULER)
    parser.add_argument("--lora", type=str, default=LORA)
    parser.add_argument("--controlnet", type=str, default=CONTROLNET)
    parser.add_argument("--steps", type=int, default=STEPS)
    parser.add_argument("--prompt", type=str, default=PROMPT)
    parser.add_argument("--negative-prompt", type=str, default=NEGATIVE_PROMPT)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--warmups", type=int, default=WARMUPS)
    parser.add_argument("--batch", type=int, default=BATCH)
    parser.add_argument("--height", nargs="+", type=int, default=HEIGHT)
    parser.add_argument("--width", nargs="+", type=int, default=WIDTH)
    parser.add_argument("--extra-call-kwargs", type=str, default=EXTRA_CALL_KWARGS)
    parser.add_argument("--input-image", type=str, default=INPUT_IMAGE)
    parser.add_argument("--control-image", type=str, default=CONTROL_IMAGE)
    parser.add_argument("--output-image", type=str, default=OUTPUT_IMAGE)
    parser.add_argument("--world-size", type=int, default=WORLD_SIZE)
    parser.add_argument("--print-output", action="store_true")
    parser.add_argument("--display-output", action="store_true")
    return parser.parse_args()


def load_pipe(
    pipeline_cls,
    pipe,
    variant=None,
    dtype=None,
    device=None,
    custom_pipeline=None,
    scheduler=None,
    lora=None,
    controlnet=None,
):
    extra_kwargs = {}
    if custom_pipeline is not None:
        extra_kwargs["custom_pipeline"] = custom_pipeline
    if variant is not None:
        extra_kwargs["variant"] = variant
    if dtype is not None:
        extra_kwargs["torch_dtype"] = dtype
    if controlnet is not None:
        from diffusers import ControlNetModel

        controlnet = ControlNetModel.from_pretrained(controlnet, torch_dtype=torch.float16)
        extra_kwargs["controlnet"] = controlnet
    pipe = pipeline_cls.from_pretrained(pipe, **extra_kwargs)
    if scheduler is not None:
        scheduler_cls = getattr(importlib.import_module("diffusers"), scheduler)
        pipe.scheduler = scheduler_cls.from_config(pipe.scheduler.config)
    if lora is not None:
        pipe.load_lora_weights(lora)
        pipe.fuse_lora()
    pipe.safety_checker = None
    if device is not None:
        pipe.to(device)
    return pipe


class IterationProfiler:
    def __init__(self, steps=None):
        self.begin = None
        self.end = None
        self.num_iterations = 0
        self.steps = steps

    def get_iter_per_sec(self):
        if self.begin is None or self.end is None:
            return None
        self.end.synchronize()
        dur = self.begin.elapsed_time(self.end)
        return self.num_iterations / dur * 1000.0

    def callback_on_step_end(self, pipe, i, t, callback_kwargs):
        if self.begin is None:
            event = torch.cuda.Event(enable_timing=True)
            event.record()
            self.begin = event
        else:
            if self.steps is None or i == self.steps - 1:
                event = torch.cuda.Event(enable_timing=True)
                event.record()
                self.end = event
            self.num_iterations += 1
        return callback_kwargs


def main():
    args = parse_args()

    if args.pipeline_class is None:
        if args.input_image is None:
            from diffusers import AutoPipelineForText2Image as pipeline_cls
        else:
            from diffusers import AutoPipelineForImage2Image as pipeline_cls
    else:
        pipeline_cls = getattr(diffusers, args.pipeline_class)

    dtype = getattr(torch, args.dtype)
    device = torch.device(args.device)

    pipe = load_pipe(
        pipeline_cls,
        args.model,
        variant=args.variant,
        dtype=dtype,
        device=device,
        custom_pipeline=args.custom_pipeline,
        scheduler=args.scheduler,
        lora=args.lora,
        controlnet=args.controlnet,
    )

    core_net = None
    if core_net is None:
        core_net = getattr(pipe, "unet", None)
    if core_net is None:
        core_net = getattr(pipe, "transformer", None)

    world_size = args.world_size
    if world_size is None or world_size != 0:
        import piflux

        if world_size is not None:
            piflux.config.world_size = world_size

        piflux.patch_pipe(pipe)

    heights = args.height
    widths = args.width
    if hasattr(core_net, "config") and hasattr(core_net.config, "sample_size") and hasattr(pipe, "vae_scale_factor"):
        if not heights:
            heights = [core_net.config.sample_size * pipe.vae_scale_factor]
        if not widths:
            widths = [core_net.config.sample_size * pipe.vae_scale_factor]
    if not heights:
        heights = [None]
    if not widths:
        widths = [None]

    torch.cuda.reset_peak_memory_stats()
    for height, width in itertools.product(heights, widths):
        print(f"Running with height={height}, width={width}")
        if args.input_image is None:
            input_image = None
        else:
            input_image = load_image(args.input_image)
            input_image = input_image.resize((width, height), Image.LANCZOS)

        if args.control_image is None:
            if args.controlnet is None:
                control_image = None
            else:
                control_image = Image.new("RGB", (width, height))
                draw = ImageDraw.Draw(control_image)
                draw.ellipse((width // 4, height // 4, width // 4 * 3, height // 4 * 3), fill=(255, 255, 255))
                del draw
        else:
            control_image = load_image(args.control_image)
            control_image = control_image.resize((width, height), Image.LANCZOS)

        def get_kwarg_inputs():
            kwarg_inputs = dict(
                prompt=args.prompt,
                num_images_per_prompt=args.batch,
                generator=None if args.seed is None else torch.Generator(device="cuda").manual_seed(args.seed),
                **(dict() if args.extra_call_kwargs is None else json.loads(args.extra_call_kwargs)),
                **(dict() if height is None else {"height": height}),
                **(dict() if width is None else {"width": width}),
            )
            if args.negative_prompt is not None:
                kwarg_inputs["negative_prompt"] = args.negative_prompt
            if args.steps is not None:
                kwarg_inputs["num_inference_steps"] = args.steps
            if input_image is not None:
                kwarg_inputs["image"] = input_image
            if control_image is not None:
                if input_image is None:
                    kwarg_inputs["image"] = control_image
                else:
                    kwarg_inputs["control_image"] = control_image
            return kwarg_inputs

        # NOTE: Warm it up.
        # The initial calls will trigger compilation and might be very slow.
        # After that, it should be very fast.
        if args.warmups > 0:
            print("Begin warmup")
            for _ in range(args.warmups):
                pipe(**get_kwarg_inputs())
            print("End warmup")

        # Let's see it!
        # Note: Progress bar might work incorrectly due to the async nature of CUDA.
        kwarg_inputs = get_kwarg_inputs()
        iter_profiler = IterationProfiler(steps=args.steps)
        if "callback_on_step_end" in inspect.signature(pipe).parameters:
            kwarg_inputs["callback_on_step_end"] = iter_profiler.callback_on_step_end
        begin = time.time()
        output_images = pipe(**kwarg_inputs).images
        end = time.time()

        if args.print_output:
            # Let's view it in terminal!
            from piflux.utils.term_image import print_image

            for image in output_images:
                print_image(image, max_width=80)
        if args.display_output:
            from piflux.utils.term_image import display_image

            for image in output_images:
                display_image(image, width="50%")

        print(f"Inference time: {end - begin:.3f}s")
        iter_per_sec = iter_profiler.get_iter_per_sec()
        if iter_per_sec is not None:
            print(f"Iterations per second: {iter_per_sec:.3f}")

    peak_mem = torch.cuda.max_memory_allocated()
    print(f"Peak memory: {peak_mem / 1024**3:.3f}GiB")

    if args.output_image is not None:
        output_images[0].save(args.output_image)
    else:
        print("Please set `--output-image` to save the output image, the terminal preview is inaccurate.")


if __name__ == "__main__":
    main()
