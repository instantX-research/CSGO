import torch
from ip_adapter.utils import BLOCKS as BLOCKS
from ip_adapter.utils import controlnet_BLOCKS as controlnet_BLOCKS
from PIL import Image
from diffusers import (
    AutoencoderKL,
    ControlNetModel,
    StableDiffusionXLControlNetPipeline,
)
from ip_adapter import CSGO

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

base_model_path = "./base_models/stable-diffusion-xl-base-1.0"  
image_encoder_path = "./base_models/IP-Adapter/sdxl_models/image_encoder"
csgo_ckpt = "./CSGO/csgo_4_32.bin"
pretrained_vae_name_or_path = './base_models/sdxl-vae-fp16-fix'
controlnet_path = "./base_models/TTPLanet_SDXL_Controlnet_Tile_Realistic"
weight_dtype = torch.float16

vae = AutoencoderKL.from_pretrained(pretrained_vae_name_or_path, torch_dtype=torch.float16)
controlnet = ControlNetModel.from_pretrained(controlnet_path, torch_dtype=torch.float16, use_safetensors=True)
pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
    base_model_path,
    controlnet=controlnet,
    torch_dtype=torch.float16,
    add_watermarker=False,
    vae=vae
)
pipe.enable_vae_tiling()

target_content_blocks = BLOCKS['content']
target_style_blocks = BLOCKS['style']
controlnet_target_content_blocks = controlnet_BLOCKS['content']
controlnet_target_style_blocks = controlnet_BLOCKS['style']

# Fixed CSGO initialization with correct parameters
csgo = CSGO(pipe, image_encoder_path, csgo_ckpt, device, 
            num_content_tokens=4,
            num_style_tokens=32,  # Your original had 32, but default is 4
            target_content_blocks=target_content_blocks, 
            target_style_blocks=target_style_blocks,
            controlnet_adapter=True,  # This parameter exists
            controlnet_target_content_blocks=controlnet_target_content_blocks, 
            controlnet_target_style_blocks=controlnet_target_style_blocks,
            content_model_resampler=True,
            style_model_resampler=True)

print("✅ CSGO initialized successfully!")

style_name = 'omi_s5.jpg'
content_name = 'omi_c6.png'
style_image = Image.open("./assets/{}".format(style_name)).convert('RGB')  # Load as PIL Image
content_image = Image.open('./assets/{}'.format(content_name)).convert('RGB')

caption = 'a small house with a sheep statue on top of it'
num_sample = 4

print("Starting image generation...")

# Create output directory
import os
os.makedirs("outputs", exist_ok=True)

# Image-driven style transfer
print("1. Image-driven style transfer...")
images = csgo.generate(pil_content_image=content_image, 
                       pil_style_image=style_image,
                       prompt=caption,
                       negative_prompt="text, watermark, lowres, low quality, worst quality, deformed, glitch, low contrast, noisy, saturation, blurry",
                       content_scale=1.0,
                       style_scale=1.0,
                       guidance_scale=10,
                       num_images_per_prompt=num_sample,
                       num_samples=1,
                       num_inference_steps=50,
                       seed=42,
                       image=content_image.convert('RGB'),
                       controlnet_conditioning_scale=0.6)

# Save the images
for i, image in enumerate(images):
    image.save(f"outputs/style_transfer_{i}.png")
print(f"✅ Generated and saved {len(images)} images for style transfer to outputs/")

# Text-driven stylized synthesis
print("2. Text-driven stylized synthesis...")
caption = 'a cat'
images = csgo.generate(pil_content_image=content_image, 
                       pil_style_image=style_image,
                       prompt=caption,
                       negative_prompt="text, watermark, lowres, low quality, worst quality, deformed, glitch, low contrast, noisy, saturation, blurry",
                       content_scale=1.0,
                       style_scale=1.0,
                       guidance_scale=10,
                       num_images_per_prompt=num_sample,
                       num_samples=1,
                       num_inference_steps=50,
                       seed=42,
                       image=content_image.convert('RGB'),
                       controlnet_conditioning_scale=0.01)

# Save the images
for i, image in enumerate(images):
    image.save(f"outputs/text_driven_cat_{i}.png")
print(f"✅ Generated and saved {len(images)} images for text-driven synthesis to outputs/")

# Text editing-driven stylized synthesis
print("3. Text editing-driven stylized synthesis...")
caption = 'a bicycle parked next to the house' # 'a small house'
images = csgo.generate(pil_content_image=content_image, 
                       pil_style_image=style_image,
                       prompt=caption,
                       negative_prompt="text, watermark, lowres, low quality, worst quality, deformed, glitch, low contrast, noisy, saturation, blurry",
                       content_scale=1.0,
                       style_scale=1.0,
                       guidance_scale=10,
                       num_images_per_prompt=num_sample,
                       num_samples=1,
                       num_inference_steps=50,
                       seed=42,
                       image=content_image.convert('RGB'),
                       controlnet_conditioning_scale=0.4)

# Save the images
for i, image in enumerate(images):
    image.save(f"outputs/text_editing_house_{i}.png")
print(f"✅ Generated and saved {len(images)} images for text editing synthesis to outputs/")

print("🎉 All generations completed successfully!")
print("📁 All images saved in the 'outputs/' directory")