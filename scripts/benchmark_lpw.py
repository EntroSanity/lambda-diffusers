import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:4096'
import torch
import csv
import itertools
from diffusers import DiffusionPipeline
from torch.utils.benchmark import Timer

device = torch.device("cuda:0")
prompt = "a photo of an astronaut riding a horse on mars"
neg_prompt = "blur, low quality, carton, animate"
num_inference_steps = 30

def get_inference_pipeline_SDXL(precision):
    """
    Returns a Hugging Face diffusion pipeline for Stable Diffusion XL without LPW.
    """
    assert precision in ("half", "single"), "Precision must be either 'half' or 'single'."
    
    pipe = DiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=torch.float32 if precision == "single" else torch.float16,
        use_safetensors=True,
        variant="fp16"
    )
    return pipe.to(device)

def get_inference_pipeline_LPW(precision):
    """
    Returns a Hugging Face diffusion pipeline for Stable Diffusion XL with LPW enabled,
    enhancing performance on long prompts.
    """
    assert precision in ("half", "single"), "Precision must be either 'half' or 'single'."
    
    pipe = DiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=torch.float32 if precision == "single" else torch.float16,
        use_safetensors=True,
        custom_pipeline="lpw_stable_diffusion_xl",
        variant="fp16"
    )
    return pipe.to(device)

def do_inference(pipe, width, height):
    torch.cuda.empty_cache()
    with torch.no_grad():
        images = pipe(prompt=prompt, negative_prompt=neg_prompt, width=width, height=height, num_inference_steps=num_inference_steps).images[0]
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
def run_benchmark(pipeline_type, precision, width, height):
    if pipeline_type == 'SDXL':
        pipe = get_inference_pipeline_SDXL(precision)
    elif pipeline_type == 'LPW':
        pipe = get_inference_pipeline_LPW(precision)
    else:
        raise ValueError("Invalid pipeline type")
    
    latency = get_inference_time(pipe, width, height)
    memory_usage = get_inference_memory(pipe, width, height)
    logs = {"pipeline_type": pipeline_type, "precision": precision, "width": width, "height": height, "latency": latency, "memory_usage": memory_usage}
    print(logs)
    print("============================")
    return logs

def get_device_description():
    return torch.cuda.get_device_name()

def run_benchmark_grid():
    device_desc = get_device_description()
    pipeline_types = ['LPW', 'SDXL']
    precision_options = ("single", "half")
    image_sizes = [(512, 512), (512, 768), (512, 1024), (1024, 1024)]

    for pipeline_type in pipeline_types:
        results_file = f"results_{pipeline_type}.csv"
        with open(results_file, "w") as f:
            writer = csv.writer(f)
            writer.writerow(["device", "pipeline_type", "precision", "width", "height", "latency", "memory_usage"])

            for precision, (width, height) in itertools.product(precision_options, image_sizes):
                try:
                    log = run_benchmark(pipeline_type, precision, width, height)
                    writer.writerow([device_desc, pipeline_type, precision, width, height, log["latency"], log["memory_usage"]])
                except Exception as e:
                    print(f"Error with {pipeline_type}: {e}")
                    writer.writerow([device_desc, pipeline_type, precision, width, height, "error", "error"])


if __name__ == "__main__":
    run_benchmark_grid()
