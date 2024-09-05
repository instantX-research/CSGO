import sys
# sys.path.append("../")
sys.path.append("./")
import gradio as gr
import torch
from ip_adapter.utils import BLOCKS as BLOCKS
from ip_adapter.utils import controlnet_BLOCKS as controlnet_BLOCKS
from ip_adapter.utils import resize_content
import cv2
import numpy as np
import random
from PIL import Image
from transformers import AutoImageProcessor, AutoModel
from diffusers import (
    AutoencoderKL,
    ControlNetModel,
    StableDiffusionXLControlNetPipeline,

)
from ip_adapter import CSGO
from transformers import BlipProcessor, BlipForConditionalGeneration

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

base_model_path = "stabilityai/stable-diffusion-xl-base-1.0"
image_encoder_path = "h94/IP-Adapter/sdxl_models/image_encoder"
csgo_ckpt ='InstantX/CSGO/csgo_4_32.bin'
pretrained_vae_name_or_path ='madebyollin/sdxl-vae-fp16-fix'
controlnet_path = "TTPlanet/TTPLanet_SDXL_Controlnet_Tile_Realistic"
weight_dtype = torch.float16




vae = AutoencoderKL.from_pretrained(pretrained_vae_name_or_path,torch_dtype=torch.float16)
controlnet = ControlNetModel.from_pretrained(controlnet_path, torch_dtype=torch.float16,use_safetensors=True)
pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
    base_model_path,
    controlnet=controlnet,
    torch_dtype=torch.float16,
    add_watermarker=False,
    vae=vae
)
pipe.enable_vae_tiling()
blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large").to(device)

target_content_blocks = BLOCKS['content']
target_style_blocks = BLOCKS['style']
controlnet_target_content_blocks = controlnet_BLOCKS['content']
controlnet_target_style_blocks = controlnet_BLOCKS['style']

csgo = CSGO(pipe, image_encoder_path, csgo_ckpt, device, num_content_tokens=4, num_style_tokens=32,
            target_content_blocks=target_content_blocks, target_style_blocks=target_style_blocks,
            controlnet_adapter=True,
            controlnet_target_content_blocks=controlnet_target_content_blocks,
            controlnet_target_style_blocks=controlnet_target_style_blocks,
            content_model_resampler=True,
            style_model_resampler=True,
            )

MAX_SEED = np.iinfo(np.int32).max

def randomize_seed_fn(seed: int, randomize_seed: bool) -> int:
    if randomize_seed:
        seed = random.randint(0, MAX_SEED)
    return seed





def get_example():
    case = [
        [
            "./assets/img_0.png",
            './assets/img_1.png',
            "Image-Driven Style Transfer",
            "there is a small house with a sheep statue on top of it",
            1.0,
            0.6,
            1.0,
        ],
        [
         None,
         './assets/img_1.png',
            "Text-Driven Style Synthesis",
         "a cat",
            1.0,
         0.01,
            1.0
         ],
        [
            None,
            './assets/img_2.png',
            "Text-Driven Style Synthesis",
            "a building",
            0.5,
            0.01,
            1.0
        ],
        [
            "./assets/img_0.png",
            './assets/img_1.png',
            "Text Edit-Driven Style Synthesis",
            "there is a small house",
            1.0,
            0.4,
            1.0
        ],
    ]
    return case


def run_for_examples(content_image_pil,style_image_pil,target, prompt,scale_c_controlnet, scale_c, scale_s):
    return create_image(
        content_image_pil=content_image_pil,
        style_image_pil=style_image_pil,
        prompt=prompt,
        scale_c_controlnet=scale_c_controlnet,
        scale_c=scale_c,
        scale_s=scale_s,
        guidance_scale=7.0,
        num_samples=3,
        num_inference_steps=50,
        seed=42,
        target=target,
    )
def randomize_seed_fn(seed: int, randomize_seed: bool) -> int:
    if randomize_seed:
        seed = random.randint(0, MAX_SEED)
    return seed

def image_grid(imgs, rows, cols):
    assert len(imgs) == rows * cols

    w, h = imgs[0].size
    grid = Image.new('RGB', size=(cols * w, rows * h))
    grid_w, grid_h = grid.size

    for i, img in enumerate(imgs):
        grid.paste(img, box=(i % cols * w, i // cols * h))
    return grid
def create_image(content_image_pil,
                 style_image_pil,
                 prompt,
                 scale_c_controlnet,
                 scale_c,
                 scale_s,
                 guidance_scale,
                 num_samples,
                 num_inference_steps,
                 seed,
                 target="Image-Driven Style Transfer",
):

    if content_image_pil is None:
        content_image_pil = Image.fromarray(
            np.zeros((1024, 1024, 3), dtype=np.uint8)).convert('RGB')

    if prompt is None or prompt == '':
        with torch.no_grad():
            inputs = blip_processor(content_image_pil, return_tensors="pt").to(device)
            out = blip_model.generate(**inputs)
            prompt = blip_processor.decode(out[0], skip_special_tokens=True)
    width, height, content_image = resize_content(content_image_pil)
    style_image = style_image_pil
    neg_content_prompt='text, watermark, lowres, low quality, worst quality, deformed, glitch, low contrast, noisy, saturation, blurry'
    if target =="Image-Driven Style Transfer":
        with torch.no_grad():
            images = csgo.generate(pil_content_image=content_image, pil_style_image=style_image,
                                   prompt=prompt,
                                   negative_prompt=neg_content_prompt,
                                   height=height,
                                   width=width,
                                   content_scale=scale_c,
                                   style_scale=scale_s,
                                   guidance_scale=guidance_scale,
                                   num_images_per_prompt=num_samples,
                                   num_inference_steps=num_inference_steps,
                                   num_samples=1,
                                   seed=seed,
                                   image=content_image.convert('RGB'),
                                   controlnet_conditioning_scale=scale_c_controlnet,
                                   )

    elif target =="Text-Driven Style Synthesis":
        content_image = Image.fromarray(
            np.zeros((1024, 1024, 3), dtype=np.uint8)).convert('RGB')
        with torch.no_grad():
            images = csgo.generate(pil_content_image=content_image, pil_style_image=style_image,
                                   prompt=prompt,
                                   negative_prompt="text, watermark, lowres, low quality, worst quality, deformed, glitch, low contrast, noisy, saturation, blurry",
                                   height=height,
                                   width=width,
                                   content_scale=scale_c,
                                   style_scale=scale_s,
                                   guidance_scale=7,
                                   num_images_per_prompt=num_samples,
                                   num_inference_steps=num_inference_steps,
                                   num_samples=1,
                                   seed=42,
                                   image=content_image.convert('RGB'),
                                   controlnet_conditioning_scale=scale_c_controlnet,
                                   )
    elif target =="Text Edit-Driven Style Synthesis":

        with torch.no_grad():
            images = csgo.generate(pil_content_image=content_image, pil_style_image=style_image,
                                   prompt=prompt,
                                   negative_prompt=neg_content_prompt,
                                   height=height,
                                   width=width,
                                   content_scale=scale_c,
                                   style_scale=scale_s,
                                   guidance_scale=guidance_scale,
                                   num_images_per_prompt=num_samples,
                                   num_inference_steps=num_inference_steps,
                                   num_samples=1,
                                   seed=seed,
                                   image=content_image.convert('RGB'),
                                   controlnet_conditioning_scale=scale_c_controlnet,
                                   )

    return [image_grid(images, 1, num_samples)]


def pil_to_cv2(image_pil):
    image_np = np.array(image_pil)
    image_cv2 = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    return image_cv2


# Description
title = r"""
<h1 align="center">CSGO: Content-Style Composition in Text-to-Image Generation</h1>
"""

description = r"""
<b>Official 🤗 Gradio demo</b> for <a href='https://github.com/instantX-research/CSGO' target='_blank'><b>CSGO: Content-Style Composition in Text-to-Image Generation</b></a>.<br>
How to use:<br>
1. Upload a content image if you want to use image-driven style transfer.
2. Upload a style image.
3. Sets the type of task to perform, by default image-driven style transfer is performed. Options are <b>Image-driven style transfer, Text-driven style synthesis, and Text editing-driven style synthesis<b>.
4. <b>If you choose a text-driven task, enter your desired prompt<b>.
5.If you don't provide a prompt, the default is to use the BLIP model to generate the caption.
6. Click the <b>Submit</b> button to begin customization.
7. Share your stylized photo with your friends and enjoy! 😊

Advanced usage:<br>
1. Click advanced options.
2. Choose different guidance and steps.
"""

article = r"""
---
📝 **Tips**
In CSGO, the more accurate the text prompts for content images, the better the content retention.
Text-driven style synthesis and text-edit-driven style synthesis are expected to be more stable in the next release.
---
📝 **Citation**
<br>
If our work is helpful for your research or applications, please cite us via:
```bibtex
@article{xing2024csgo,
       title={CSGO: Content-Style Composition in Text-to-Image Generation}, 
       author={Peng Xing and Haofan Wang and Yanpeng Sun and Qixun Wang and Xu Bai and Hao Ai and Renyuan Huang and Zechao Li},
       year={2024},
       journal = {arXiv 2408.16766},
}
```
📧 **Contact**
<br>
If you have any questions, please feel free to open an issue or directly reach us out at <b>xingp_ng@njust.edu.cn</b>.
"""

block = gr.Blocks(css="footer {visibility: hidden}").queue(max_size=10, api_open=False)
with block:
    # description
    gr.Markdown(title)
    gr.Markdown(description)

    with gr.Tabs():
        with gr.Row():
            with gr.Column():
                with gr.Row():
                    with gr.Column():
                        content_image_pil = gr.Image(label="Content Image (optional)", type='pil')
                        style_image_pil = gr.Image(label="Style Image", type='pil')

                target = gr.Radio(["Image-Driven Style Transfer", "Text-Driven Style Synthesis", "Text Edit-Driven Style Synthesis"],
                                  value="Image-Driven Style Transfer",
                                  label="task")

                prompt = gr.Textbox(label="Prompt",
                                    value="there is a small house with a sheep statue on top of it")

                scale_c_controlnet = gr.Slider(minimum=0, maximum=2.0, step=0.01, value=0.6,
                                               label="Content Scale for controlnet")
                scale_c = gr.Slider(minimum=0, maximum=2.0, step=0.01, value=0.6, label="Content Scale for IPA")

                scale_s = gr.Slider(minimum=0, maximum=2.0, step=0.01, value=1.0, label="Style Scale")
                with gr.Accordion(open=False, label="Advanced Options"):

                    guidance_scale = gr.Slider(minimum=1, maximum=15.0, step=0.01, value=7.0, label="guidance scale")
                    num_samples = gr.Slider(minimum=1, maximum=4.0, step=1.0, value=1.0, label="num samples")
                    num_inference_steps = gr.Slider(minimum=5, maximum=100.0, step=1.0, value=50,
                                                    label="num inference steps")
                    seed = gr.Slider(minimum=-1000000, maximum=1000000, value=1, step=1, label="Seed Value")
                    randomize_seed = gr.Checkbox(label="Randomize seed", value=True)

                generate_button = gr.Button("Generate Image")

            with gr.Column():
                generated_image = gr.Gallery(label="Generated Image")

        generate_button.click(
            fn=randomize_seed_fn,
            inputs=[seed, randomize_seed],
            outputs=seed,
            queue=False,
            api_name=False,
        ).then(
            fn=create_image,
            inputs=[content_image_pil,
                    style_image_pil,
                    prompt,
                    scale_c_controlnet,
                    scale_c,
                    scale_s,
                    guidance_scale,
                    num_samples,
                    num_inference_steps,
                    seed,
                    target,],
            outputs=[generated_image])

    gr.Examples(
        examples=get_example(),
        inputs=[content_image_pil,style_image_pil,target, prompt,scale_c_controlnet, scale_c, scale_s],
        fn=run_for_examples,
        outputs=[generated_image],
        cache_examples=True,
    )

    gr.Markdown(article)

block.launch(server_name="0.0.0.0", server_port=1234)
