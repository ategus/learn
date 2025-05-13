
import torch
from diffusers import FluxPipeline

pipe = FluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-schnell",
    torch_dtype=torch.float16
)

pipe.to("cpu")

prompt = "A cat holding a sign that says hello world"
image = pipe(
    prompt=prompt,
    guidance_scale=0.0,
    height=256,
    width=256,
    num_inference_steps=4,
).images[0]

image.save("image_cpu.png")
