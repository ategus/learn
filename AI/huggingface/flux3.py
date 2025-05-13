import torch
from diffusers import FluxPipeline

# 1. Load the FLUX.1-schnell model with FP16 weights
pipe = FluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-schnell",
    torch_dtype=torch.float16
)

# 2. Enable more aggressive CPU offloading
#    This offloads parts of the model from GPU to CPU when not in use.
pipe.enable_sequential_cpu_offload()

# 3. Enable attention slicing to reduce peak GPU memory usage
pipe.enable_attention_slicing()

# 4. (Optional) If you have xformers installed for memory-efficient attention:
# pip install xformers
# pipe.enable_xformers_memory_efficient_attention()

prompt = "A cat holding a sign that says hello world"

# 5. Lower resolution from 768×1360 to something smaller (e.g., 512×512)
out = pipe(
    prompt=prompt,
    guidance_scale=0.0,  # 0 means basically no classifier-free guidance
    height=512,
    width=512,
    num_inference_steps=4,     # Few inference steps to reduce load
    max_sequence_length=256,   # Just as you had before
).images[0]

out.save("image.png")

