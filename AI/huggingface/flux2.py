
import torch
from diffusers import FluxPipeline

pipe = FluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-schnell",
    torch_dtype=torch.float16
)

# Optional: more advanced offloading or memory saving
# pipe.enable_attention_slicing()
# pipe.enable_xformers_memory_efficient_attention()  # requires xformers installed
pipe.enable_sequential_cpu_offload()

prompt = "A cat holding a sign that says hello world"

out = pipe(
    prompt=prompt,
    guidance_scale=0.,
    height=512,   # <--- smaller resolution
    width=512,    # <--- smaller resolution
    num_inference_steps=4,
    max_sequence_length=256,
).images[0]

out.save("image.png")
