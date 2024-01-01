import os
import torch
import csv
import itertools
from diffusers import StableDiffusionPipeline
from torch.utils.benchmark import Timer

device = torch.device("cuda:0")
prompt = "a photo of an astronaut riding a horse on mars"
num_inference_steps = 30

def get_inference_pipeline(precision):
    """
    Returns HuggingFace diffuser pipeline
    """
    assert precision in ("half", "single"), "precision in ['half', 'single']"

    pipe = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        revision="main" if precision == "single" else "fp16",
        torch_dtype=torch.float32 if precision == "single" else torch.float16,
    )
    return pipe.to(device)

def do_inference(pipe, width, height):
    torch.cuda.empty_cache()
    images = pipe(prompt=prompt, num_inference_steps=num_inference_steps, width=width, height=height).images
    return images

def get_inference_time(pipe, width, height):
    timer = Timer(
        stmt="do_inference(pipe, width, height)",
        setup="from __main__ import do_inference",
        globals={"pipe": pipe, "width": width, "height": height},
    )
    profile_result = timer.timeit(1)
    return round(profile_result.mean, 2)

def get_inference_memory(pipe, width, height):
    do_inference(pipe, width, height)
    mem = torch.cuda.memory_reserved(device=device)
    return round(mem / 1e9, 2)

@torch.inference_mode()
def run_benchmark(precision, width, height):
    pipe = get_inference_pipeline(precision)
    latency = get_inference_time(pipe, width, height)
    memory_usage = get_inference_memory(pipe, width, height)
    logs = {"precision": precision, "width": width, "height": height, "latency": latency, "memory_usage": memory_usage}
    print(logs)
    print("============================")
    return logs

def get_device_description():
    return torch.cuda.get_device_name()

def run_benchmark_grid():
    device_desc = get_device_description()
    precision_options = ("single", "half")
    image_sizes = [(512, 512), (512, 768), (512, 1024), (1024, 1024)]

    with open("results.csv", "w") as f:
        writer = csv.writer(f)
        writer.writerow(["device", "precision", "width", "height", "latency", "memory_usage"])

        for precision, (width, height) in itertools.product(precision_options, image_sizes):
            try:
                log = run_benchmark(precision, width, height)
                writer.writerow([device_desc, precision, width, height, log["latency"], log["memory_usage"]])
            except Exception as e:
                print(f"Error: {e}")
                writer.writerow([device_desc, precision, width, height, "error", "error"])

if __name__ == "__main__":
    run_benchmark_grid()
