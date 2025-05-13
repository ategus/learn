import torch
from diffusers import FluxPipeline

pipe = FluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-schnell",
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True  # Sometimes helps reduce peak loading memory
)

# Most aggressive CPU offloading:
pipe.enable_sequential_cpu_offload()

# Slicing to reduce memory spikes in attention:
pipe.enable_attention_slicing()

# If you installed xformers: pip install xformers
# pipe.enable_xformers_memory_efficient_attention()

prompt = "A cat holding a sign that says hello world"

out = pipe(
    prompt=prompt,
    guidance_scale=0.0,
    height=256,
    width=256,
    num_inference_steps=4,
    max_sequence_length=256,
).images[0]

out.save("image.png")

