import os
from typing import List

import torch
from diffusers import StableDiffusionPipeline
from diffusers.pipelines.controlnet import MultiControlNetModel
from PIL import Image
from safetensors import safe_open
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection
from torchvision import transforms
from .utils import is_torch2_available, get_generator

# import torchvision.transforms.functional as Func

# from .clip_style_models import CSD_CLIP, convert_state_dict

if is_torch2_available():
    from .attention_processor import (
        AttnProcessor2_0 as AttnProcessor,
    )
    from .attention_processor import (
        CNAttnProcessor2_0 as CNAttnProcessor,
    )
    from .attention_processor import (
        IPAttnProcessor2_0 as IPAttnProcessor,
    )
    from .attention_processor import IP_CS_AttnProcessor2_0 as IP_CS_AttnProcessor
else:
    from .attention_processor import AttnProcessor, CNAttnProcessor, IPAttnProcessor
from .resampler import Resampler

from transformers import AutoImageProcessor, AutoModel


class ImageProjModel(torch.nn.Module):
    """Projection Model"""

    def __init__(self, cross_attention_dim=1024, clip_embeddings_dim=1024, clip_extra_context_tokens=4):
        super().__init__()

        self.generator = None
        self.cross_attention_dim = cross_attention_dim
        self.clip_extra_context_tokens = clip_extra_context_tokens
        # print(clip_embeddings_dim, self.clip_extra_context_tokens, cross_attention_dim)
        self.proj = torch.nn.Linear(clip_embeddings_dim, self.clip_extra_context_tokens * cross_attention_dim)
        self.norm = torch.nn.LayerNorm(cross_attention_dim)

    def forward(self, image_embeds):
        embeds = image_embeds
        clip_extra_context_tokens = self.proj(embeds).reshape(
            -1, self.clip_extra_context_tokens, self.cross_attention_dim
        )
        clip_extra_context_tokens = self.norm(clip_extra_context_tokens)
        return clip_extra_context_tokens


class MLPProjModel(torch.nn.Module):
    """SD model with image prompt"""

    def __init__(self, cross_attention_dim=1024, clip_embeddings_dim=1024):
        super().__init__()

        self.proj = torch.nn.Sequential(
            torch.nn.Linear(clip_embeddings_dim, clip_embeddings_dim),
            torch.nn.GELU(),
            torch.nn.Linear(clip_embeddings_dim, cross_attention_dim),
            torch.nn.LayerNorm(cross_attention_dim)
        )

    def forward(self, image_embeds):
        clip_extra_context_tokens = self.proj(image_embeds)
        return clip_extra_context_tokens


class IPAdapter:
    def __init__(self, sd_pipe, image_encoder_path, ip_ckpt, device, num_tokens=4, target_blocks=["block"]):
        self.device = device
        self.image_encoder_path = image_encoder_path
        self.ip_ckpt = ip_ckpt
        self.num_tokens = num_tokens
        self.target_blocks = target_blocks

        self.pipe = sd_pipe.to(self.device)
        self.set_ip_adapter()

        # load image encoder
        self.image_encoder = CLIPVisionModelWithProjection.from_pretrained(self.image_encoder_path).to(
            self.device, dtype=torch.float16
        )
        self.clip_image_processor = CLIPImageProcessor()
        # image proj model
        self.image_proj_model = self.init_proj()

        self.load_ip_adapter()

    def init_proj(self):
        image_proj_model = ImageProjModel(
            cross_attention_dim=self.pipe.unet.config.cross_attention_dim,
            clip_embeddings_dim=self.image_encoder.config.projection_dim,
            clip_extra_context_tokens=self.num_tokens,
        ).to(self.device, dtype=torch.float16)
        return image_proj_model

    def set_ip_adapter(self):
        unet = self.pipe.unet
        attn_procs = {}
        for name in unet.attn_processors.keys():
            cross_attention_dim = None if name.endswith("attn1.processor") else unet.config.cross_attention_dim
            if name.startswith("mid_block"):
                hidden_size = unet.config.block_out_channels[-1]
            elif name.startswith("up_blocks"):
                block_id = int(name[len("up_blocks.")])
                hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
            elif name.startswith("down_blocks"):
                block_id = int(name[len("down_blocks.")])
                hidden_size = unet.config.block_out_channels[block_id]
            if cross_attention_dim is None:
                attn_procs[name] = AttnProcessor()
            else:
                selected = False
                for block_name in self.target_blocks:
                    if block_name in name:
                        selected = True
                        break
                if selected:
                    attn_procs[name] = IPAttnProcessor(
                        hidden_size=hidden_size,
                        cross_attention_dim=cross_attention_dim,
                        scale=1.0,
                        num_tokens=self.num_tokens,
                    ).to(self.device, dtype=torch.float16)
                else:
                    attn_procs[name] = IPAttnProcessor(
                        hidden_size=hidden_size,
                        cross_attention_dim=cross_attention_dim,
                        scale=1.0,
                        num_tokens=self.num_tokens,
                        skip=True
                    ).to(self.device, dtype=torch.float16)
        unet.set_attn_processor(attn_procs)
        if hasattr(self.pipe, "controlnet"):
            if isinstance(self.pipe.controlnet, MultiControlNetModel):
                for controlnet in self.pipe.controlnet.nets:
                    controlnet.set_attn_processor(CNAttnProcessor(num_tokens=self.num_tokens))
            else:
                self.pipe.controlnet.set_attn_processor(CNAttnProcessor(num_tokens=self.num_tokens))

    def load_ip_adapter(self):
        if os.path.splitext(self.ip_ckpt)[-1] == ".safetensors":
            state_dict = {"image_proj": {}, "ip_adapter": {}}
            with safe_open(self.ip_ckpt, framework="pt", device="cpu") as f:
                for key in f.keys():
                    if key.startswith("image_proj."):
                        state_dict["image_proj"][key.replace("image_proj.", "")] = f.get_tensor(key)
                    elif key.startswith("ip_adapter."):
                        state_dict["ip_adapter"][key.replace("ip_adapter.", "")] = f.get_tensor(key)
        else:
            state_dict = torch.load(self.ip_ckpt, map_location="cpu")
        self.image_proj_model.load_state_dict(state_dict["image_proj"])
        ip_layers = torch.nn.ModuleList(self.pipe.unet.attn_processors.values())
        ip_layers.load_state_dict(state_dict["ip_adapter"], strict=False)

    @torch.inference_mode()
    def get_image_embeds(self, pil_image=None, clip_image_embeds=None, content_prompt_embeds=None):
        if pil_image is not None:
            if isinstance(pil_image, Image.Image):
                pil_image = [pil_image]
            clip_image = self.clip_image_processor(images=pil_image, return_tensors="pt").pixel_values
            clip_image_embeds = self.image_encoder(clip_image.to(self.device, dtype=torch.float16)).image_embeds
        else:
            clip_image_embeds = clip_image_embeds.to(self.device, dtype=torch.float16)

        if content_prompt_embeds is not None:
            clip_image_embeds = clip_image_embeds - content_prompt_embeds

        image_prompt_embeds = self.image_proj_model(clip_image_embeds)
        uncond_image_prompt_embeds = self.image_proj_model(torch.zeros_like(clip_image_embeds))
        return image_prompt_embeds, uncond_image_prompt_embeds

    def set_scale(self, scale):
        for attn_processor in self.pipe.unet.attn_processors.values():
            if isinstance(attn_processor, IPAttnProcessor):
                attn_processor.scale = scale

    def generate(
            self,
            pil_image=None,
            clip_image_embeds=None,
            prompt=None,
            negative_prompt=None,
            scale=1.0,
            num_samples=4,
            seed=None,
            guidance_scale=7.5,
            num_inference_steps=30,
            neg_content_emb=None,
            **kwargs,
    ):
        self.set_scale(scale)

        if pil_image is not None:
            num_prompts = 1 if isinstance(pil_image, Image.Image) else len(pil_image)
        else:
            num_prompts = clip_image_embeds.size(0)

        if prompt is None:
            prompt = "best quality, high quality"
        if negative_prompt is None:
            negative_prompt = "monochrome, lowres, bad anatomy, worst quality, low quality"

        if not isinstance(prompt, List):
            prompt = [prompt] * num_prompts
        if not isinstance(negative_prompt, List):
            negative_prompt = [negative_prompt] * num_prompts

        image_prompt_embeds, uncond_image_prompt_embeds = self.get_image_embeds(
            pil_image=pil_image, clip_image_embeds=clip_image_embeds, content_prompt_embeds=neg_content_emb
        )
        bs_embed, seq_len, _ = image_prompt_embeds.shape
        image_prompt_embeds = image_prompt_embeds.repeat(1, num_samples, 1)
        image_prompt_embeds = image_prompt_embeds.view(bs_embed * num_samples, seq_len, -1)
        uncond_image_prompt_embeds = uncond_image_prompt_embeds.repeat(1, num_samples, 1)
        uncond_image_prompt_embeds = uncond_image_prompt_embeds.view(bs_embed * num_samples, seq_len, -1)

        with torch.inference_mode():
            prompt_embeds_, negative_prompt_embeds_ = self.pipe.encode_prompt(
                prompt,
                device=self.device,
                num_images_per_prompt=num_samples,
                do_classifier_free_guidance=True,
                negative_prompt=negative_prompt,
            )
            prompt_embeds = torch.cat([prompt_embeds_, image_prompt_embeds], dim=1)
            negative_prompt_embeds = torch.cat([negative_prompt_embeds_, uncond_image_prompt_embeds], dim=1)

        generator = get_generator(seed, self.device)

        images = self.pipe(
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            generator=generator,
            **kwargs,
        ).images

        return images


class IPAdapter_CS:
    def __init__(self, sd_pipe, image_encoder_path, ip_ckpt, device, num_content_tokens=4,
                 num_style_tokens=4,
                 target_content_blocks=["block"], target_style_blocks=["block"], content_image_encoder_path=None,
                  controlnet_adapter=False,
                 controlnet_target_content_blocks=None,
                 controlnet_target_style_blocks=None,
                 content_model_resampler=False,
                 style_model_resampler=False,
                ):
        self.device = device
        self.image_encoder_path = image_encoder_path
        self.ip_ckpt = ip_ckpt
        self.num_content_tokens = num_content_tokens
        self.num_style_tokens = num_style_tokens
        self.content_target_blocks = target_content_blocks
        self.style_target_blocks = target_style_blocks

        self.content_model_resampler = content_model_resampler
        self.style_model_resampler = style_model_resampler

        self.controlnet_adapter = controlnet_adapter
        self.controlnet_target_content_blocks = controlnet_target_content_blocks
        self.controlnet_target_style_blocks = controlnet_target_style_blocks

        self.pipe = sd_pipe.to(self.device)
        self.set_ip_adapter()
        self.content_image_encoder_path = content_image_encoder_path


        # load image encoder
        if content_image_encoder_path is not None:
            self.content_image_encoder = AutoModel.from_pretrained(content_image_encoder_path).to(self.device,
                                                                                                  dtype=torch.float16)
            self.content_image_processor = AutoImageProcessor.from_pretrained(content_image_encoder_path)
        else:
            self.content_image_encoder = CLIPVisionModelWithProjection.from_pretrained(self.image_encoder_path).to(
                self.device, dtype=torch.float16
            )
            self.content_image_processor = CLIPImageProcessor()
        # model.requires_grad_(False)

        self.image_encoder = CLIPVisionModelWithProjection.from_pretrained(self.image_encoder_path).to(
            self.device, dtype=torch.float16
        )
        # if self.use_CSD is not None:
        #     self.style_image_encoder = CSD_CLIP("vit_large", "default",self.use_CSD+"/ViT-L-14.pt")
        #     model_path = self.use_CSD+"/checkpoint.pth"
        #     checkpoint = torch.load(model_path, map_location="cpu")
        #     state_dict = convert_state_dict(checkpoint['model_state_dict'])
        #     self.style_image_encoder.load_state_dict(state_dict, strict=False)
        #
        #     normalize = transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
        #     self.style_preprocess = transforms.Compose([
        #         transforms.Resize(size=224, interpolation=Func.InterpolationMode.BICUBIC),
        #         transforms.CenterCrop(224),
        #         transforms.ToTensor(),
        #         normalize,
        #     ])

        self.clip_image_processor = CLIPImageProcessor()
        # image proj model
        self.content_image_proj_model = self.init_proj(self.num_content_tokens, content_or_style_='content',
                                                       model_resampler=self.content_model_resampler)
        self.style_image_proj_model = self.init_proj(self.num_style_tokens, content_or_style_='style',
                                                     model_resampler=self.style_model_resampler)

        self.load_ip_adapter()

    def init_proj(self, num_tokens, content_or_style_='content', model_resampler=False):

        # print('@@@@',self.pipe.unet.config.cross_attention_dim,self.image_encoder.config.projection_dim)
        if content_or_style_ == 'content' and self.content_image_encoder_path is not None:
            image_proj_model = ImageProjModel(
                cross_attention_dim=self.pipe.unet.config.cross_attention_dim,
                clip_embeddings_dim=self.content_image_encoder.config.projection_dim,
                clip_extra_context_tokens=num_tokens,
            ).to(self.device, dtype=torch.float16)
            return image_proj_model

        image_proj_model = ImageProjModel(
            cross_attention_dim=self.pipe.unet.config.cross_attention_dim,
            clip_embeddings_dim=self.image_encoder.config.projection_dim,
            clip_extra_context_tokens=num_tokens,
        ).to(self.device, dtype=torch.float16)
        return image_proj_model

    def set_ip_adapter(self):
        unet = self.pipe.unet
        attn_procs = {}
        for name in unet.attn_processors.keys():
            cross_attention_dim = None if name.endswith("attn1.processor") else unet.config.cross_attention_dim
            if name.startswith("mid_block"):
                hidden_size = unet.config.block_out_channels[-1]
            elif name.startswith("up_blocks"):
                block_id = int(name[len("up_blocks.")])
                hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
            elif name.startswith("down_blocks"):
                block_id = int(name[len("down_blocks.")])
                hidden_size = unet.config.block_out_channels[block_id]
            if cross_attention_dim is None:
                attn_procs[name] = AttnProcessor()
            else:
                # layername_id += 1
                selected = False
                for block_name in self.style_target_blocks:
                    if block_name in name:
                        selected = True
                        # print(name)
                        attn_procs[name] = IP_CS_AttnProcessor(
                            hidden_size=hidden_size,
                            cross_attention_dim=cross_attention_dim,
                            style_scale=1.0,
                            style=True,
                            num_content_tokens=self.num_content_tokens,
                            num_style_tokens=self.num_style_tokens,
                        )
                for block_name in self.content_target_blocks:
                    if block_name in name:
                        # selected = True
                        if selected is False:
                            attn_procs[name] = IP_CS_AttnProcessor(
                                hidden_size=hidden_size,
                                cross_attention_dim=cross_attention_dim,
                                content_scale=1.0,
                                content=True,
                                num_content_tokens=self.num_content_tokens,
                                num_style_tokens=self.num_style_tokens,
                            )
                        else:
                            attn_procs[name].set_content_ipa(content_scale=1.0)
                            # attn_procs[name].content=True

                if selected is False:
                    attn_procs[name] = IP_CS_AttnProcessor(
                        hidden_size=hidden_size,
                        cross_attention_dim=cross_attention_dim,
                        num_content_tokens=self.num_content_tokens,
                        num_style_tokens=self.num_style_tokens,
                        skip=True,
                    )

                attn_procs[name].to(self.device, dtype=torch.float16)
        unet.set_attn_processor(attn_procs)
        if hasattr(self.pipe, "controlnet"):
            if self.controlnet_adapter is False:
                if isinstance(self.pipe.controlnet, MultiControlNetModel):
                    for controlnet in self.pipe.controlnet.nets:
                        controlnet.set_attn_processor(CNAttnProcessor(
                            num_tokens=self.num_content_tokens + self.num_style_tokens))
                else:
                    self.pipe.controlnet.set_attn_processor(CNAttnProcessor(
                        num_tokens=self.num_content_tokens + self.num_style_tokens))

            else:
                controlnet_attn_procs = {}
                controlnet_style_target_blocks = self.controlnet_target_style_blocks
                controlnet_content_target_blocks = self.controlnet_target_content_blocks
                for name in self.pipe.controlnet.attn_processors.keys():
                    # print(name)
                    cross_attention_dim = None if name.endswith(
                        "attn1.processor") else self.pipe.controlnet.config.cross_attention_dim
                    if name.startswith("mid_block"):
                        hidden_size = self.pipe.controlnet.config.block_out_channels[-1]
                    elif name.startswith("up_blocks"):
                        block_id = int(name[len("up_blocks.")])
                        hidden_size = list(reversed(self.pipe.controlnet.config.block_out_channels))[block_id]
                    elif name.startswith("down_blocks"):
                        block_id = int(name[len("down_blocks.")])
                        hidden_size = self.pipe.controlnet.config.block_out_channels[block_id]
                    if cross_attention_dim is None:
                        # layername_id += 1
                        controlnet_attn_procs[name] = AttnProcessor()

                    else:
                        # layername_id += 1
                        selected = False
                        for block_name in controlnet_style_target_blocks:
                            if block_name in name:
                                selected = True
                                # print(name)
                                controlnet_attn_procs[name] = IP_CS_AttnProcessor(
                                    hidden_size=hidden_size,
                                    cross_attention_dim=cross_attention_dim,
                                    style_scale=1.0,
                                    style=True,
                                    num_content_tokens=self.num_content_tokens,
                                    num_style_tokens=self.num_style_tokens,
                                )

                        for block_name in controlnet_content_target_blocks:
                            if block_name in name:
                                if selected is False:
                                    controlnet_attn_procs[name] = IP_CS_AttnProcessor(
                                        hidden_size=hidden_size,
                                        cross_attention_dim=cross_attention_dim,
                                        content_scale=1.0,
                                        content=True,
                                        num_content_tokens=self.num_content_tokens,
                                        num_style_tokens=self.num_style_tokens,
                                    )

                                    selected = True
                                elif selected is True:
                                    controlnet_attn_procs[name].set_content_ipa(content_scale=1.0)

                                # if args.content_image_encoder_type !='dinov2':
                                #     weights = {
                                #         "to_k_ip.weight": state_dict["ip_adapter"][str(layername_id) + ".to_k_ip.weight"],
                                #         "to_v_ip.weight": state_dict["ip_adapter"][str(layername_id) + ".to_v_ip.weight"],
                                #     }
                                #     attn_procs[name].load_state_dict(weights)
                        if selected is False:
                            controlnet_attn_procs[name] = IP_CS_AttnProcessor(
                                hidden_size=hidden_size,
                                cross_attention_dim=cross_attention_dim,
                                num_content_tokens=self.num_content_tokens,
                                num_style_tokens=self.num_style_tokens,
                                skip=True,
                            )
                        controlnet_attn_procs[name].to(self.device, dtype=torch.float16)
                        # layer_name = name.split(".processor")[0]
                        # # print(state_dict["ip_adapter"].keys())
                        # weights = {
                        #     "to_k_ip.weight": state_dict["ip_adapter"][str(layername_id) + ".to_k_ip.weight"],
                        #     "to_v_ip.weight": state_dict["ip_adapter"][str(layername_id) + ".to_v_ip.weight"],
                        # }
                        # attn_procs[name].load_state_dict(weights)
                self.pipe.controlnet.set_attn_processor(controlnet_attn_procs)

    def load_ip_adapter(self):
        if os.path.splitext(self.ip_ckpt)[-1] == ".safetensors":
            state_dict = {"content_image_proj": {}, "style_image_proj": {}, "ip_adapter": {}}
            with safe_open(self.ip_ckpt, framework="pt", device="cpu") as f:
                for key in f.keys():
                    if key.startswith("content_image_proj."):
                        state_dict["content_image_proj"][key.replace("content_image_proj.", "")] = f.get_tensor(key)
                    elif key.startswith("style_image_proj."):
                        state_dict["style_image_proj"][key.replace("style_image_proj.", "")] = f.get_tensor(key)
                    elif key.startswith("ip_adapter."):
                        state_dict["ip_adapter"][key.replace("ip_adapter.", "")] = f.get_tensor(key)
        else:
            state_dict = torch.load(self.ip_ckpt, map_location="cpu")
        self.content_image_proj_model.load_state_dict(state_dict["content_image_proj"])
        self.style_image_proj_model.load_state_dict(state_dict["style_image_proj"])

        if 'conv_in_unet_sd' in state_dict.keys():
            self.pipe.unet.conv_in.load_state_dict(state_dict["conv_in_unet_sd"], strict=True)
        ip_layers = torch.nn.ModuleList(self.pipe.unet.attn_processors.values())
        ip_layers.load_state_dict(state_dict["ip_adapter"], strict=False)

        if self.controlnet_adapter is True:
            print('loading controlnet_adapter')
            self.pipe.controlnet.load_state_dict(state_dict["controlnet_adapter_modules"], strict=False)

    @torch.inference_mode()
    def get_image_embeds(self, pil_image=None, clip_image_embeds=None, content_prompt_embeds=None,
                         content_or_style_=''):
        # if pil_image is not None:
        #     if isinstance(pil_image, Image.Image):
        #         pil_image = [pil_image]
        #     clip_image = self.clip_image_processor(images=pil_image, return_tensors="pt").pixel_values
        #     clip_image_embeds = self.image_encoder(clip_image.to(self.device, dtype=torch.float16)).image_embeds
        # else:
        #     clip_image_embeds = clip_image_embeds.to(self.device, dtype=torch.float16)

        # if content_prompt_embeds is not None:
        #     clip_image_embeds = clip_image_embeds - content_prompt_embeds

        if content_or_style_ == 'content':
            if pil_image is not None:
                if isinstance(pil_image, Image.Image):
                    pil_image = [pil_image]
                if self.content_image_proj_model is not None:
                    clip_image = self.content_image_processor(images=pil_image, return_tensors="pt").pixel_values
                    clip_image_embeds = self.content_image_encoder(
                        clip_image.to(self.device, dtype=torch.float16)).image_embeds
                else:
                    clip_image = self.clip_image_processor(images=pil_image, return_tensors="pt").pixel_values
                    clip_image_embeds = self.image_encoder(clip_image.to(self.device, dtype=torch.float16)).image_embeds
            else:
                clip_image_embeds = clip_image_embeds.to(self.device, dtype=torch.float16)

            image_prompt_embeds = self.content_image_proj_model(clip_image_embeds)
            uncond_image_prompt_embeds = self.content_image_proj_model(torch.zeros_like(clip_image_embeds))
            return image_prompt_embeds, uncond_image_prompt_embeds
        if content_or_style_ == 'style':
            if pil_image is not None:
                if self.use_CSD is not None:
                    clip_image = self.style_preprocess(pil_image).unsqueeze(0).to(self.device, dtype=torch.float32)
                    clip_image_embeds = self.style_image_encoder(clip_image)
                else:
                    if isinstance(pil_image, Image.Image):
                        pil_image = [pil_image]
                    clip_image = self.clip_image_processor(images=pil_image, return_tensors="pt").pixel_values
                    clip_image_embeds = self.image_encoder(clip_image.to(self.device, dtype=torch.float16)).image_embeds


            else:
                clip_image_embeds = clip_image_embeds.to(self.device, dtype=torch.float16)
            image_prompt_embeds = self.style_image_proj_model(clip_image_embeds)
            uncond_image_prompt_embeds = self.style_image_proj_model(torch.zeros_like(clip_image_embeds))
            return image_prompt_embeds, uncond_image_prompt_embeds

    def set_scale(self, content_scale, style_scale):
        for attn_processor in self.pipe.unet.attn_processors.values():
            if isinstance(attn_processor, IP_CS_AttnProcessor):
                if attn_processor.content is True:
                    attn_processor.content_scale = content_scale

                if attn_processor.style is True:
                    attn_processor.style_scale = style_scale
                    # print('style_scale:',style_scale)
        if self.controlnet_adapter is not None:
            for attn_processor in self.pipe.controlnet.attn_processors.values():

                if isinstance(attn_processor, IP_CS_AttnProcessor):
                    if attn_processor.content is True:
                        attn_processor.content_scale = content_scale
                        # print(content_scale)

                    if attn_processor.style is True:
                        attn_processor.style_scale = style_scale

    def generate(
            self,
            pil_content_image=None,
            pil_style_image=None,
            clip_content_image_embeds=None,
            clip_style_image_embeds=None,
            prompt=None,
            negative_prompt=None,
            content_scale=1.0,
            style_scale=1.0,
            num_samples=4,
            seed=None,
            guidance_scale=7.5,
            num_inference_steps=30,
            neg_content_emb=None,
            **kwargs,
    ):
        self.set_scale(content_scale, style_scale)

        if pil_content_image is not None:
            num_prompts = 1 if isinstance(pil_content_image, Image.Image) else len(pil_content_image)
        else:
            num_prompts = clip_content_image_embeds.size(0)

        if prompt is None:
            prompt = "best quality, high quality"
        if negative_prompt is None:
            negative_prompt = "monochrome, lowres, bad anatomy, worst quality, low quality"

        if not isinstance(prompt, List):
            prompt = [prompt] * num_prompts
        if not isinstance(negative_prompt, List):
            negative_prompt = [negative_prompt] * num_prompts

        content_image_prompt_embeds, uncond_content_image_prompt_embeds = self.get_image_embeds(
            pil_image=pil_content_image, clip_image_embeds=clip_content_image_embeds
        )
        style_image_prompt_embeds, uncond_style_image_prompt_embeds = self.get_image_embeds(
            pil_image=pil_style_image, clip_image_embeds=clip_style_image_embeds
        )

        bs_embed, seq_len, _ = content_image_prompt_embeds.shape
        content_image_prompt_embeds = content_image_prompt_embeds.repeat(1, num_samples, 1)
        content_image_prompt_embeds = content_image_prompt_embeds.view(bs_embed * num_samples, seq_len, -1)
        uncond_content_image_prompt_embeds = uncond_content_image_prompt_embeds.repeat(1, num_samples, 1)
        uncond_content_image_prompt_embeds = uncond_content_image_prompt_embeds.view(bs_embed * num_samples, seq_len,
                                                                                     -1)

        bs_style_embed, seq_style_len, _ = content_image_prompt_embeds.shape
        style_image_prompt_embeds = style_image_prompt_embeds.repeat(1, num_samples, 1)
        style_image_prompt_embeds = style_image_prompt_embeds.view(bs_embed * num_samples, seq_style_len, -1)
        uncond_style_image_prompt_embeds = uncond_style_image_prompt_embeds.repeat(1, num_samples, 1)
        uncond_style_image_prompt_embeds = uncond_style_image_prompt_embeds.view(bs_embed * num_samples, seq_style_len,
                                                                                 -1)

        with torch.inference_mode():
            prompt_embeds_, negative_prompt_embeds_ = self.pipe.encode_prompt(
                prompt,
                device=self.device,
                num_images_per_prompt=num_samples,
                do_classifier_free_guidance=True,
                negative_prompt=negative_prompt,
            )
            prompt_embeds = torch.cat([prompt_embeds_, content_image_prompt_embeds, style_image_prompt_embeds], dim=1)
            negative_prompt_embeds = torch.cat([negative_prompt_embeds_,
                                                uncond_content_image_prompt_embeds, uncond_style_image_prompt_embeds],
                                               dim=1)

        generator = get_generator(seed, self.device)

        images = self.pipe(
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            generator=generator,
            **kwargs,
        ).images

        return images


class IPAdapterXL_CS(IPAdapter_CS):
    """SDXL"""

    def generate(
            self,
            pil_content_image,
            pil_style_image,
            prompt=None,
            negative_prompt=None,
            content_scale=1.0,
            style_scale=1.0,
            num_samples=4,
            seed=None,
            content_image_embeds=None,
            style_image_embeds=None,
            num_inference_steps=30,
            neg_content_emb=None,
            neg_content_prompt=None,
            neg_content_scale=1.0,
            **kwargs,
    ):
        self.set_scale(content_scale, style_scale)

        num_prompts = 1 if isinstance(pil_content_image, Image.Image) else len(pil_content_image)

        if prompt is None:
            prompt = "best quality, high quality"
        if negative_prompt is None:
            negative_prompt = "monochrome, lowres, bad anatomy, worst quality, low quality"

        if not isinstance(prompt, List):
            prompt = [prompt] * num_prompts
        if not isinstance(negative_prompt, List):
            negative_prompt = [negative_prompt] * num_prompts

        content_image_prompt_embeds, uncond_content_image_prompt_embeds = self.get_image_embeds(pil_content_image,
                                                                                                content_image_embeds,
                                                                                                content_or_style_='content')



        style_image_prompt_embeds, uncond_style_image_prompt_embeds = self.get_image_embeds(pil_style_image,
                                                                                            style_image_embeds,
                                                                                            content_or_style_='style')

        bs_embed, seq_len, _ = content_image_prompt_embeds.shape

        content_image_prompt_embeds = content_image_prompt_embeds.repeat(1, num_samples, 1)
        content_image_prompt_embeds = content_image_prompt_embeds.view(bs_embed * num_samples, seq_len, -1)

        uncond_content_image_prompt_embeds = uncond_content_image_prompt_embeds.repeat(1, num_samples, 1)
        uncond_content_image_prompt_embeds = uncond_content_image_prompt_embeds.view(bs_embed * num_samples, seq_len,
                                                                                     -1)
        bs_style_embed, seq_style_len, _ = style_image_prompt_embeds.shape
        style_image_prompt_embeds = style_image_prompt_embeds.repeat(1, num_samples, 1)
        style_image_prompt_embeds = style_image_prompt_embeds.view(bs_embed * num_samples, seq_style_len, -1)
        uncond_style_image_prompt_embeds = uncond_style_image_prompt_embeds.repeat(1, num_samples, 1)
        uncond_style_image_prompt_embeds = uncond_style_image_prompt_embeds.view(bs_embed * num_samples, seq_style_len,
                                                                                 -1)

        with torch.inference_mode():
            (
                prompt_embeds,
                negative_prompt_embeds,
                pooled_prompt_embeds,
                negative_pooled_prompt_embeds,
            ) = self.pipe.encode_prompt(
                prompt,
                num_images_per_prompt=num_samples,
                do_classifier_free_guidance=True,
                negative_prompt=negative_prompt,
            )
            prompt_embeds = torch.cat([prompt_embeds, content_image_prompt_embeds, style_image_prompt_embeds], dim=1)
            negative_prompt_embeds = torch.cat([negative_prompt_embeds,
                                                uncond_content_image_prompt_embeds, uncond_style_image_prompt_embeds],
                                               dim=1)

        self.generator = get_generator(seed, self.device)

        images = self.pipe(
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
            num_inference_steps=num_inference_steps,
            generator=self.generator,
            **kwargs,
        ).images
        return images


class CSGO(IPAdapterXL_CS):
    """SDXL"""

    def init_proj(self, num_tokens, content_or_style_='content', model_resampler=False):
        if content_or_style_ == 'content':
            if model_resampler:
                image_proj_model = Resampler(
                    dim=self.pipe.unet.config.cross_attention_dim,
                    depth=4,
                    dim_head=64,
                    heads=12,
                    num_queries=num_tokens,
                    embedding_dim=self.content_image_encoder.config.hidden_size,
                    output_dim=self.pipe.unet.config.cross_attention_dim,
                    ff_mult=4,
                ).to(self.device, dtype=torch.float16)
            else:
                image_proj_model = ImageProjModel(
                    cross_attention_dim=self.pipe.unet.config.cross_attention_dim,
                    clip_embeddings_dim=self.image_encoder.config.projection_dim,
                    clip_extra_context_tokens=num_tokens,
                ).to(self.device, dtype=torch.float16)
        if content_or_style_ == 'style':
            if model_resampler:
                image_proj_model = Resampler(
                    dim=self.pipe.unet.config.cross_attention_dim,
                    depth=4,
                    dim_head=64,
                    heads=12,
                    num_queries=num_tokens,
                    embedding_dim=self.content_image_encoder.config.hidden_size,
                    output_dim=self.pipe.unet.config.cross_attention_dim,
                    ff_mult=4,
                ).to(self.device, dtype=torch.float16)
            else:
                image_proj_model = ImageProjModel(
                    cross_attention_dim=self.pipe.unet.config.cross_attention_dim,
                    clip_embeddings_dim=self.image_encoder.config.projection_dim,
                    clip_extra_context_tokens=num_tokens,
                ).to(self.device, dtype=torch.float16)
        return image_proj_model

    @torch.inference_mode()
    def get_image_embeds(self, pil_image=None, clip_image_embeds=None, content_or_style_=''):
        if isinstance(pil_image, Image.Image):
            pil_image = [pil_image]
        if content_or_style_ == 'style':

            if self.style_model_resampler:
                clip_image = self.clip_image_processor(images=pil_image, return_tensors="pt").pixel_values
                clip_image_embeds = self.image_encoder(clip_image.to(self.device, dtype=torch.float16),
                                                       output_hidden_states=True).hidden_states[-2]
                image_prompt_embeds = self.style_image_proj_model(clip_image_embeds)
                uncond_image_prompt_embeds = self.style_image_proj_model(torch.zeros_like(clip_image_embeds))
            else:


                clip_image = self.clip_image_processor(images=pil_image, return_tensors="pt").pixel_values
                clip_image_embeds = self.image_encoder(clip_image.to(self.device, dtype=torch.float16)).image_embeds
                image_prompt_embeds = self.style_image_proj_model(clip_image_embeds)
                uncond_image_prompt_embeds = self.style_image_proj_model(torch.zeros_like(clip_image_embeds))
            return image_prompt_embeds, uncond_image_prompt_embeds


        else:

            if self.content_image_encoder_path is not None:
                clip_image = self.content_image_processor(images=pil_image, return_tensors="pt").pixel_values
                outputs = self.content_image_encoder(clip_image.to(self.device, dtype=torch.float16),
                                                     output_hidden_states=True)
                clip_image_embeds = outputs.last_hidden_state
                image_prompt_embeds = self.content_image_proj_model(clip_image_embeds)

                # uncond_clip_image_embeds = self.image_encoder(
                #     torch.zeros_like(clip_image), output_hidden_states=True
                # ).last_hidden_state
                uncond_image_prompt_embeds = self.content_image_proj_model(torch.zeros_like(clip_image_embeds))
                return image_prompt_embeds, uncond_image_prompt_embeds

            else:
                if self.content_model_resampler:

                    clip_image = self.clip_image_processor(images=pil_image, return_tensors="pt").pixel_values

                    clip_image = clip_image.to(self.device, dtype=torch.float16)
                    clip_image_embeds = self.image_encoder(clip_image, output_hidden_states=True).hidden_states[-2]
                    # clip_image_embeds = clip_image_embeds.to(self.device, dtype=torch.float16)
                    image_prompt_embeds = self.content_image_proj_model(clip_image_embeds)
                    # uncond_clip_image_embeds = self.image_encoder(
                    #             torch.zeros_like(clip_image), output_hidden_states=True
                    #         ).hidden_states[-2]
                    uncond_image_prompt_embeds = self.content_image_proj_model(torch.zeros_like(clip_image_embeds))
                else:
                    clip_image = self.clip_image_processor(images=pil_image, return_tensors="pt").pixel_values
                    clip_image_embeds = self.image_encoder(clip_image.to(self.device, dtype=torch.float16)).image_embeds
                    image_prompt_embeds = self.content_image_proj_model(clip_image_embeds)
                    uncond_image_prompt_embeds = self.content_image_proj_model(torch.zeros_like(clip_image_embeds))

                return image_prompt_embeds, uncond_image_prompt_embeds

        #     # clip_image = self.clip_image_processor(images=pil_image, return_tensors="pt").pixel_values
        #     clip_image = clip_image.to(self.device, dtype=torch.float16)
        #     clip_image_embeds = self.image_encoder(clip_image, output_hidden_states=True).hidden_states[-2]
        #     image_prompt_embeds = self.content_image_proj_model(clip_image_embeds)
        #     uncond_clip_image_embeds = self.image_encoder(
        #         torch.zeros_like(clip_image), output_hidden_states=True
        #     ).hidden_states[-2]
        #     uncond_image_prompt_embeds = self.content_image_proj_model(uncond_clip_image_embeds)
        # return image_prompt_embeds, uncond_image_prompt_embeds


class IPAdapterXL(IPAdapter):
    """SDXL"""

    def generate(
            self,
            pil_image,
            prompt=None,
            negative_prompt=None,
            scale=1.0,
            num_samples=4,
            seed=None,
            num_inference_steps=30,
            neg_content_emb=None,
            neg_content_prompt=None,
            neg_content_scale=1.0,
            **kwargs,
    ):
        self.set_scale(scale)

        num_prompts = 1 if isinstance(pil_image, Image.Image) else len(pil_image)

        if prompt is None:
            prompt = "best quality, high quality"
        if negative_prompt is None:
            negative_prompt = "monochrome, lowres, bad anatomy, worst quality, low quality"

        if not isinstance(prompt, List):
            prompt = [prompt] * num_prompts
        if not isinstance(negative_prompt, List):
            negative_prompt = [negative_prompt] * num_prompts

        if neg_content_emb is None:
            if neg_content_prompt is not None:
                with torch.inference_mode():
                    (
                        prompt_embeds_,  # torch.Size([1, 77, 2048])
                        negative_prompt_embeds_,
                        pooled_prompt_embeds_,  # torch.Size([1, 1280])
                        negative_pooled_prompt_embeds_,
                    ) = self.pipe.encode_prompt(
                        neg_content_prompt,
                        num_images_per_prompt=num_samples,
                        do_classifier_free_guidance=True,
                        negative_prompt=negative_prompt,
                    )
                    pooled_prompt_embeds_ *= neg_content_scale
            else:
                pooled_prompt_embeds_ = neg_content_emb
        else:
            pooled_prompt_embeds_ = None

        image_prompt_embeds, uncond_image_prompt_embeds = self.get_image_embeds(pil_image,
                                                                                content_prompt_embeds=pooled_prompt_embeds_)
        bs_embed, seq_len, _ = image_prompt_embeds.shape
        image_prompt_embeds = image_prompt_embeds.repeat(1, num_samples, 1)
        image_prompt_embeds = image_prompt_embeds.view(bs_embed * num_samples, seq_len, -1)
        uncond_image_prompt_embeds = uncond_image_prompt_embeds.repeat(1, num_samples, 1)
        uncond_image_prompt_embeds = uncond_image_prompt_embeds.view(bs_embed * num_samples, seq_len, -1)

        with torch.inference_mode():
            (
                prompt_embeds,
                negative_prompt_embeds,
                pooled_prompt_embeds,
                negative_pooled_prompt_embeds,
            ) = self.pipe.encode_prompt(
                prompt,
                num_images_per_prompt=num_samples,
                do_classifier_free_guidance=True,
                negative_prompt=negative_prompt,
            )
            prompt_embeds = torch.cat([prompt_embeds, image_prompt_embeds], dim=1)
            negative_prompt_embeds = torch.cat([negative_prompt_embeds, uncond_image_prompt_embeds], dim=1)

        self.generator = get_generator(seed, self.device)

        images = self.pipe(
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
            num_inference_steps=num_inference_steps,
            generator=self.generator,
            **kwargs,
        ).images

        return images


class IPAdapterPlus(IPAdapter):
    """IP-Adapter with fine-grained features"""

    def init_proj(self):
        image_proj_model = Resampler(
            dim=self.pipe.unet.config.cross_attention_dim,
            depth=4,
            dim_head=64,
            heads=12,
            num_queries=self.num_tokens,
            embedding_dim=self.image_encoder.config.hidden_size,
            output_dim=self.pipe.unet.config.cross_attention_dim,
            ff_mult=4,
        ).to(self.device, dtype=torch.float16)
        return image_proj_model

    @torch.inference_mode()
    def get_image_embeds(self, pil_image=None, clip_image_embeds=None):
        if isinstance(pil_image, Image.Image):
            pil_image = [pil_image]
        clip_image = self.clip_image_processor(images=pil_image, return_tensors="pt").pixel_values
        clip_image = clip_image.to(self.device, dtype=torch.float16)
        clip_image_embeds = self.image_encoder(clip_image, output_hidden_states=True).hidden_states[-2]
        image_prompt_embeds = self.image_proj_model(clip_image_embeds)
        uncond_clip_image_embeds = self.image_encoder(
            torch.zeros_like(clip_image), output_hidden_states=True
        ).hidden_states[-2]
        uncond_image_prompt_embeds = self.image_proj_model(uncond_clip_image_embeds)
        return image_prompt_embeds, uncond_image_prompt_embeds


class IPAdapterFull(IPAdapterPlus):
    """IP-Adapter with full features"""

    def init_proj(self):
        image_proj_model = MLPProjModel(
            cross_attention_dim=self.pipe.unet.config.cross_attention_dim,
            clip_embeddings_dim=self.image_encoder.config.hidden_size,
        ).to(self.device, dtype=torch.float16)
        return image_proj_model


class IPAdapterPlusXL(IPAdapter):
    """SDXL"""

    def init_proj(self):
        image_proj_model = Resampler(
            dim=1280,
            depth=4,
            dim_head=64,
            heads=20,
            num_queries=self.num_tokens,
            embedding_dim=self.image_encoder.config.hidden_size,
            output_dim=self.pipe.unet.config.cross_attention_dim,
            ff_mult=4,
        ).to(self.device, dtype=torch.float16)
        return image_proj_model

    @torch.inference_mode()
    def get_image_embeds(self, pil_image):
        if isinstance(pil_image, Image.Image):
            pil_image = [pil_image]
        clip_image = self.clip_image_processor(images=pil_image, return_tensors="pt").pixel_values
        clip_image = clip_image.to(self.device, dtype=torch.float16)
        clip_image_embeds = self.image_encoder(clip_image, output_hidden_states=True).hidden_states[-2]
        image_prompt_embeds = self.image_proj_model(clip_image_embeds)
        uncond_clip_image_embeds = self.image_encoder(
            torch.zeros_like(clip_image), output_hidden_states=True
        ).hidden_states[-2]
        uncond_image_prompt_embeds = self.image_proj_model(uncond_clip_image_embeds)
        return image_prompt_embeds, uncond_image_prompt_embeds

    def generate(
            self,
            pil_image,
            prompt=None,
            negative_prompt=None,
            scale=1.0,
            num_samples=4,
            seed=None,
            num_inference_steps=30,
            **kwargs,
    ):
        self.set_scale(scale)

        num_prompts = 1 if isinstance(pil_image, Image.Image) else len(pil_image)

        if prompt is None:
            prompt = "best quality, high quality"
        if negative_prompt is None:
            negative_prompt = "monochrome, lowres, bad anatomy, worst quality, low quality"

        if not isinstance(prompt, List):
            prompt = [prompt] * num_prompts
        if not isinstance(negative_prompt, List):
            negative_prompt = [negative_prompt] * num_prompts

        image_prompt_embeds, uncond_image_prompt_embeds = self.get_image_embeds(pil_image)
        bs_embed, seq_len, _ = image_prompt_embeds.shape
        image_prompt_embeds = image_prompt_embeds.repeat(1, num_samples, 1)
        image_prompt_embeds = image_prompt_embeds.view(bs_embed * num_samples, seq_len, -1)
        uncond_image_prompt_embeds = uncond_image_prompt_embeds.repeat(1, num_samples, 1)
        uncond_image_prompt_embeds = uncond_image_prompt_embeds.view(bs_embed * num_samples, seq_len, -1)

        with torch.inference_mode():
            (
                prompt_embeds,
                negative_prompt_embeds,
                pooled_prompt_embeds,
                negative_pooled_prompt_embeds,
            ) = self.pipe.encode_prompt(
                prompt,
                num_images_per_prompt=num_samples,
                do_classifier_free_guidance=True,
                negative_prompt=negative_prompt,
            )
            prompt_embeds = torch.cat([prompt_embeds, image_prompt_embeds], dim=1)
            negative_prompt_embeds = torch.cat([negative_prompt_embeds, uncond_image_prompt_embeds], dim=1)

        generator = get_generator(seed, self.device)

        images = self.pipe(
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
            num_inference_steps=num_inference_steps,
            generator=generator,
            **kwargs,
        ).images

        return images
