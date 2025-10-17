import os
import zipfile
import aiohttp
import asyncio

import requests
import json
import base64
import torch
import numpy as np
from PIL import Image
from io import BytesIO
from typing import Optional, List, Union, Dict, Any

ucPreset_list = [
    "blurry, lowres, error, film grain, scan artifacts, worst quality, bad quality, jpeg artifacts, very displeasing, chromatic aberration, logo, dated, signature, multiple views, gigantic breasts",
    "blurry, lowres, error, worst quality, bad quality, jpeg artifacts, very displeasing, logo, dated, signature",
    ""
]


async def post_novelai(url, data, header, proxy):
    async with aiohttp.ClientSession(headers=header) as session:
        try:
            async with session.post(url, json=data, proxy=proxy) as response:
                if response.status == 429:
                    resp = await response.json()
                    message = resp.get("message", "Too many requests, retrying...")
                    await asyncio.sleep(5)
                    return await post_novelai(url, data, header, proxy)
                elif response.status == 200:
                    return await response.read()
                else:
                    raise Exception(f"Request failed with status {response.status}")
        except Exception as e:
            print(f"Error during request: {e}")
            raise e


class CharacterPrompt:
    def __init__(self, prompt: str, uc: str, x: float, y: float):
        self.prompt = prompt
        self.uc = uc
        self.x = x
        self.y = y
        self.center = {"x": self.x, "y": self.y}


class BaseRequest:
    def __init__(
            self,
            prompt: Optional[str] = "",
            negative_prompt: Optional[str] = "",
            seed: int = -1,
            sampler_name: str = "Euler a",
            batch_size: int = 1,
            n_iter: int = 1,
            steps: int = 28,
            cfg_scale: float = 7,
            width: int = 832,
            height: int = 1216,
            denoising_strength: float = 1,
            scheduler: str = "Automatic",
            send_images: bool = True,
            save_images: bool = True,
            override_settings: Dict[str, Any] = {},
            override_settings_restore_afterwards: bool = False
    ):
        self.prompt = prompt
        self.negative_prompt = negative_prompt
        self.seed = seed
        self.sampler_name = sampler_name
        self.batch_size = batch_size
        self.n_iter = n_iter
        self.steps = steps
        self.cfg_scale = cfg_scale
        self.width = width
        self.height = height
        self.denoising_strength = denoising_strength
        self.scheduler = scheduler
        self.send_images = send_images
        self.save_images = save_images
        self.override_settings = override_settings
        self.override_settings_restore_afterwards = override_settings_restore_afterwards


class NovelAITXT2IMGPayload(BaseRequest):
    def __init__(
            self, ucPreset: int,
            cfg_rescale: int, characterPrompts: list[CharacterPrompt]|list,
            prefer_brownian: bool, base_request: BaseRequest, model,
            skip_cfg_above_sigma_enable: bool = False
    ):
        super().__init__(**vars(base_request))

        self.scale = base_request.cfg_scale
        self.sampler = base_request.sampler_name
        self.n_samples = base_request.n_iter
        self.ucPreset = ucPreset
        self.cfg_rescale = cfg_rescale
        self.noise_schedule = base_request.scheduler
        self.characterPrompts = characterPrompts
        self.model = model
        self.skip_cfg_above_sigma_enable = skip_cfg_above_sigma_enable
        
        # 设置skip_cfg_above_sigma值
        if self.skip_cfg_above_sigma_enable:
            pixel_count = self.width * self.height
            if pixel_count == 1024 * 1024:
                self.skip_cfg_above_sigma = 59.04722600415217
            elif pixel_count == 1216 * 832:
                self.skip_cfg_above_sigma = 58.0
            elif pixel_count == 1408 * 704:
                self.skip_cfg_above_sigma = 57.40995413696215
            else:
                self.skip_cfg_above_sigma = 58.0
        else:
            self.skip_cfg_above_sigma = None
            
        characterPrompts_list = []

        for cp in self.characterPrompts:
            cp = dict(vars(cp))
            del cp['x']
            del cp['y']
            characterPrompts_list.append(cp)

        self.characterPrompts = characterPrompts_list
        char_captions = []
        for cp in self.characterPrompts:
            char_captions.append({"char_caption": cp["prompt"], "centers": [cp["center"]]})
        self.v4_prompt = {
            "caption": {
                "base_caption": self.prompt,
                "char_captions": char_captions
            }
        }
        char_captions = []
        for cp in self.characterPrompts:
            char_captions.append({"char_caption": cp["uc"], "centers": [cp["center"]]})
        self.v4_negative_prompt = {
            "caption": {
                "base_caption": self.negative_prompt,
                "char_captions": char_captions
            }
        }
        self.prefer_brownian = prefer_brownian


class TXT2IMGRequestExtend(BaseRequest):

    def __init__(self, script_name: str = "",
                 restore_faces: bool = False, tiling: bool = False, subseed: int = -1, subseed_strength: float = 0,
                 styles: List[str] = [], sampler_index: str = "Euler a", script_args: List[Any] = [],
                 alwayson_scripts: Dict[str, Any] = {}, hr_scale: float = 2, hr_upscaler: str = "",
                 hr_second_pass_steps: int = 10, hr_resize_x: int = 0, hr_resize_y: int = 0,
                 hr_checkpoint_name: str = "", hr_sampler_name: str = "", hr_prompt: str = "",
                 hr_negative_prompt: str = "", s_min_uncond: float = 0, s_churn: float = 0, s_tmax: float = 0,
                 s_tmin: float = 0, s_noise: float = 0, refiner_checkpoint: str = "",
                 refiner_switch_at: int = 0, disable_extra_networks: bool = False, comments: Dict[str, Any] = {},
                 enable_hr: bool = False, firstphase_width: int = 0, firstphase_height: int = 0,
                 do_not_save_samples: bool = False, do_not_save_grid: bool = False, eta: float = 0,
                 seed_resize_from_h: int = -1, seed_resize_from_w: int = -1):

        super().__init__()

        self.script_name = script_name
        self.restore_faces = restore_faces
        self.tiling = tiling
        self.subseed = subseed
        self.subseed_strength = subseed_strength
        self.styles = styles
        self.sampler_index = sampler_index
        self.script_args = script_args
        self.alwayson_scripts = alwayson_scripts
        self.hr_scale = hr_scale
        self.hr_upscaler = hr_upscaler
        self.hr_second_pass_steps = hr_second_pass_steps
        self.hr_resize_x = hr_resize_x
        self.hr_resize_y = hr_resize_y
        self.hr_checkpoint_name = hr_checkpoint_name
        self.hr_sampler_name = hr_sampler_name
        self.hr_prompt = hr_prompt
        self.hr_negative_prompt = hr_negative_prompt
        self.s_min_uncond = s_min_uncond
        self.s_churn = s_churn
        self.s_tmax = s_tmax
        self.s_tmin = s_tmin
        self.s_noise = s_noise
        self.refiner_checkpoint = refiner_checkpoint
        self.refiner_switch_at = refiner_switch_at
        self.disable_extra_networks = disable_extra_networks
        self.comments = comments
        self.enable_hr = enable_hr
        self.firstphase_width = firstphase_width
        self.firstphase_height = firstphase_height
        self.do_not_save_samples = do_not_save_samples
        self.do_not_save_grid = do_not_save_grid
        self.eta = eta
        self.seed_resize_from_h = seed_resize_from_h
        self.seed_resize_from_w = seed_resize_from_w


def tensor_to_pil(img_tensor, batch_index=0):
    # Takes an image in a batch in the form of a tensor of shape [batch_size, channels, height, width]
    # and returns an PIL Image with the corresponding mode deduced by the number of channels

    # Take the image in the batch given by batch_index
    img_tensor = img_tensor[batch_index].unsqueeze(0)
    i = 255. * img_tensor.cpu().numpy()
    img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8).squeeze())
    return img


def pil_to_tensor(image):
    # Takes a PIL image and returns a tensor of shape [1, height, width, channels]
    image = np.array(image).astype(np.float32) / 255.0
    image = torch.from_numpy(image).unsqueeze(0)
    if len(image.shape) == 3:  # If the image is grayscale, add a channel dimension
        image = image.unsqueeze(-1)
    return image


class NovelAIRequest:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "payload": ("NovelAITXT2IMGPayload",),
            },
            "optional": {
                "token": ("STRING", {"default": "ey...."}),
                "proxy": ("STRING", {"default": "http://127.0.0.1:7890"})
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "novelai_generate_image"
    OUTPUT_NODE = True
    CATEGORY = "SDWebUI-API/SDWebUI-API"

    def novelai_generate_image(self, payload, token, proxy):

        save_path = "./temp/api_request"

        if not os.path.exists(save_path):
            os.makedirs(save_path)

        token = token if token else os.getenv("NOVELAI_TOKEN", "")
        header = {
            "authorization": "Bearer " + token,
            ":authority": "https://api.novelai.net",
            ":path": "/ai/generate-image",
            "content-type": "application/json",
            "referer": "https://novelai.net",
            "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/106.0.0.0 Safari/537.36",
        }

        payload = {
            "input": payload.prompt,
            "model": payload.model,
            "parameters": dict(vars(payload))

        }

        response_data = asyncio.run(post_novelai("https://image.novelai.net/ai/generate-image", payload, header, proxy))

        with zipfile.ZipFile(BytesIO(response_data)) as z:
            z.extractall(save_path)

        image_bytes = []

        def images_to_base64(save_path):

            for filename in os.listdir(save_path):
                if filename.endswith('.png'):
                    file_path = os.path.join(save_path, filename)
                    with open(file_path, "rb") as image_file:
                        image_data = image_file.read()
                        image_bytes.append(image_data)

        images_to_base64(save_path)

        def pil_and_tensor(img_data: bytes):
            img = Image.open(BytesIO(img_data))
            tensor_img = pil_to_tensor(img)

            return tensor_img

        tensor_images = [pil_and_tensor(img_data) for img_data in image_bytes]

        tensor = torch.cat(tensor_images, dim=0)

        return (tensor,)


class CharacterPromptSelect:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "character1": ("STRING", {"default": "firefly_(honkai:_star_rail)"}),
                "character1_uc": ("STRING", {"default": "negative_prompt"}),
                "character1_x": ("FLOAT", {"default": 0.5}),
                "character1_y": ("FLOAT", {"default": 0.5}),
            },
            "optional": {
                "character2_enable": ("BOOLEAN", {"default": True}),
                "character2": ("STRING", {"default": "kafka__(honkai:_star_rail)"}),
                "character2_uc": ("STRING", {"default": "negative_prompt"}),
                "character2_x": ("FLOAT", {"default": 0.1}),
                "character2_y": ("FLOAT", {"default": 0.5}),
                "character3_enable": ("BOOLEAN", {"default": False}),
                "character3": ("STRING", {"default": ""}),
                "character3_uc": ("STRING", {"default": "negative_prompt"}),
                "character3_x": ("FLOAT", {"default": 0.1}),
                "character3_y": ("FLOAT", {"default": 0.5}),
                "character4_enable": ("BOOLEAN", {"default": False}),
                "character4": ("STRING", {"default": ""}),
                "character4_uc": ("STRING", {"default": "negative_prompt"}),
                "character4_x": ("FLOAT", {"default": 0.1}),
                "character4_y": ("FLOAT", {"default": 0.5}),
                "character5_enable": ("BOOLEAN", {"default": False}),
                "character5": ("STRING", {"default": ""}),
                "character5_uc": ("STRING", {"default": "negative_prompt"}),
                "character5_x": ("FLOAT", {"default": 0.1}),
                "character5_y": ("FLOAT", {"default": 0.5}),
            },
        }

    RETURN_TYPES = ("LIST",)
    RETURN_NAMES = ("CharacterPrompt",)
    FUNCTION = "build_character_prompt"
    OUTPUT_NODE = False
    CATEGORY = "SDWebUI-API/SDWebUI-API"

    def build_character_prompt(
        self, character1, character1_uc, character1_x, character1_y,
        character2_enable, character2, character2_uc, character2_x, character2_y, 
        character3_enable, character3, character3_uc, character3_x, character3_y, 
        character4_enable, character4, character4_uc, character4_x, character4_y, 
        character5_enable, character5, character5_uc, character5_x, character5_y
   ):
        character1_pos = {"x": character1_x, "y": character1_y}
        character2_pos = {"x": character2_x, "y": character2_y}
        character3_pos = {"x": character3_x, "y": character3_y}
        character4_pos = {"x": character4_x, "y": character4_y}
        character5_pos = {"x": character5_x, "y": character5_y}

        character_prompts = []

        character_prompts.append(CharacterPrompt(character1, character1_uc, character1_pos["x"], character1_pos["y"]))

        if character2_enable:
            character_prompts.append(CharacterPrompt(character2, character2_uc, character2_pos["x"], character2_pos["y"]))
        if character3_enable:
            character_prompts.append(CharacterPrompt(character3, character3_uc, character3_pos["x"], character3_pos["y"]))
        if character4_enable:
            character_prompts.append(CharacterPrompt(character4, character4_uc, character4_pos["x"], character4_pos["y"]))
        if character5_enable:
            character_prompts.append(CharacterPrompt(character5, character5_uc, character5_pos["x"], character5_pos["y"]))

        return (character_prompts,)


class NovelAIRequestPayload:

    @classmethod
    def INPUT_TYPES(cls):
        sampler_list = [
            "k_dpmpp_2m", "k_dpmpp_sde", "k_dpmpp_2m_sde", "k_dpmpp_2s_ancestral",
            "k_euler_ancestral", "k_euler", "ddim_v3"
        ]
        scheduler_list = ["karras"]

        model_list = [
            "nai-diffusion-4-5-full",
            "nai-diffusion-4-5-curated",
            "nai-diffusion-4-full",
            "nai-diffusion-4-curated-preview",
            "nai-diffusion-3",
            "nai-diffusion-furry-3",
            "nai-diffusion-2",
        ]

        return {
            "required": {
                "prompt": ("STRING", {"default": "prompt here"})
            },
            "optional": {
                "negative_prompt": ("STRING", {"default": "negative prompt here"}),
                "seed": ("INT", {"default": -1}),
                "sampler": (sampler_list,),
                "n_iter": ("INT", {"default": 1}),
                "steps": ("INT", {"default": 28}),
                "cfg_scale": ("FLOAT", {"default": 6}),
                "width": ("INT", {"default": 832}),
                "height": ("INT", {"default": 1216}),

                "scheduler": (scheduler_list,),
                "ucPreset": (ucPreset_list,),
                "cfg_rescale": ("INT", {"default": 0}),
                "characterPrompts": ("LIST",),
                "prefer_brownian": ("BOOLEAN", {"default": False}),
                "model": (model_list, ),
                "skip_cfg_above_sigma_enable": ("BOOLEAN", {"default": False}),

            }
        }

    RETURN_TYPES = ("NovelAITXT2IMGPayload", "DICT")
    RETURN_NAMES = ("payload", "payload_dict")
    FUNCTION = "build_payload"
    CATEGORY = "SDWebUI-API/SDWebUI-API"
    # INPUT_IS_LIST = False
    # OUTPUT_IS_LIST = (False,)

    def build_payload(
            self,
            prompt,
            negative_prompt,
            seed,
            sampler,
            n_iter,
            steps,
            cfg_scale,
            width,
            height,
            scheduler,
            ucPreset,
            cfg_rescale,
            prefer_brownian,
            model,
            characterPrompts=[],
            skip_cfg_above_sigma_enable: bool = False
    ):
        negative_prompt = negative_prompt + "," + ucPreset
        instance_ = BaseRequest(
            prompt=prompt,
            negative_prompt=negative_prompt,
            seed=seed,
            sampler_name=sampler,
            n_iter=n_iter,
            steps=steps,
            cfg_scale=cfg_scale,
            width=width,
            height=height,
            scheduler=scheduler,
        )

        ucPreset = ucPreset_list.index(ucPreset)

        instance_ = NovelAITXT2IMGPayload(ucPreset,cfg_rescale,characterPrompts,prefer_brownian, instance_, model, skip_cfg_above_sigma_enable)

        return instance_, dict(vars(instance_))


class SDWebUIRequest:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "payload": ("BaseRequest",),
                "backend_url": ("STRING", {"default": "http://127.0.0.1:7860"}),
            },
            "optional": {
                "payload_extend": ("TXT2IMGRequestExtend",)
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "sdweb_generate_image"
    OUTPUT_NODE = True
    CATEGORY = "SDWebUI-API/SDWebUI-API"

    def sdweb_generate_image(self, payload, backend_url):
        response = requests.post(backend_url + "/sdapi/v1/txt2img", json=dict(vars(payload)))
        resp = response.json()
        b64_images = resp["images"]

        def b64_to_pil_and_tensor(b64_image: str):
            img_data = base64.b64decode(b64_image)
            img = Image.open(BytesIO(img_data))
            tensor_img = pil_to_tensor(img)

            return tensor_img

        tensor_images = [b64_to_pil_and_tensor(b64_image) for b64_image in b64_images]

        tensor = torch.cat(tensor_images, dim=0)

        return (tensor,)


class SDWebUIRequestPayload:

    @classmethod
    def INPUT_TYPES(cls):
        sampler_list = [
            "DPM++ 2M", "DPM++ SDE", "DPM++ 2M SDE", "DPM++ 2M SDE Heun", "DPM++ 2S a", "DPM++ 3M SDE",
            "Euler a", "Euler", "LMS", "Heun", "DPM2", "DPM2 a", "DPM fast", "DPM adaptive", "Restart",
            "HeunPP2", "IPNDM", "IPNDM_V", "DEIS", "DDIM", "DDIM CFG++", "PLMS", "UniPC", "LCM", "DDPM"
        ]
        scheduler_list = ["Automatic", "Karras", "Exponential", "SGM Uniform", "Simple", "Normal", "DDIM", "Beta"]

        return {
            "required": {
                "prompt": ("STRING", {"default": "prompt here"})
            },
            "optional": {
                "negative_prompt": ("STRING", {"default": "negative prompt here"}),
                "seed": ("INT", {"default": -1}),
                "sampler": (sampler_list,),
                "batch_size": ("INT", {"default": 1}),
                "n_iter": ("INT", {"default": 1}),
                "steps": ("INT", {"default": 20}),
                "cfg_scale": ("FLOAT", {"default": 7}),
                "width": ("INT", {"default": 512}),
                "height": ("INT", {"default": 512}),
                "denoising_strength": ("FLOAT", {"default": 1}),
                "scheduler": (scheduler_list,),
                "send_images": ("BOOLEAN", {"default": True}),
                "save_images": ("BOOLEAN", {"default": True}),
                "override_settings": ("STRING", {"default": ""}),
                "override_settings_restore_afterwards": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("BaseRequest", "DICT")
    RETURN_NAMES = ("payload", "payload_dict")
    FUNCTION = "build_payload"
    CATEGORY = "SDWebUI-API/SDWebUI-API"
    # INPUT_IS_LIST = False
    # OUTPUT_IS_LIST = (False,)

    def build_payload(
            self,
            prompt,
            negative_prompt,
            seed,
            sampler,
            batch_size,
            n_iter,
            steps,
            cfg_scale,
            width,
            height,
            denoising_strength,
            scheduler,
            send_images,
            save_images,
            override_settings,
            override_settings_restore_afterwards
    ):
        instance_ = BaseRequest(
            prompt=prompt,
            negative_prompt=negative_prompt,
            seed=seed,
            sampler_name=sampler,
            batch_size=batch_size,
            n_iter=n_iter,
            steps=steps,
            cfg_scale=cfg_scale,
            width=width,
            height=height,
            denoising_strength=denoising_strength,
            scheduler=scheduler,
            send_images=send_images,
            save_images=save_images,
            override_settings=json.loads(override_settings),
            override_settings_restore_afterwards=override_settings_restore_afterwards

        )

        return instance_, dict(vars(instance_))


class SDWebUIRequestPayloadExtend:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "optional": {
                "script_name": ("STRING", {"default": ""}),
                "restore_faces": ("BOOLEAN", {"default": False}),
                "tiling": ("BOOLEAN", {"default": False}),
                "subseed": ("INT", {"default": -1}),
                "subseed_strength": ("FLOAT", {"default": 0}),
                "styles": ("LIST", {"default": []}),
                "sampler_index": ("STRING", {"default": ""}),
                "script_args": ("STRING", {"default": ""}),
                "alwayson_scripts": ("STRING", {"default": "alwayson_scripts"}),
                "hr_scale": ("FLOAT", {"default": 2}),
                "hr_upscaler": ("STRING", {"default": "hr_upscaler"}),
                "hr_second_pass_steps": ("INT", {"default": 0}),
                "hr_resize_x": ("INT", {"default": 0}),
                "hr_resize_y": ("INT", {"default": 0}),
                "hr_checkpoint_name": ("STRING", {"default": ""}),
                "hr_sampler_name": ("STRING", {"default": ""}),
                "hr_prompt": ("STRING", {"default": "hr_prompt"}),
                "hr_negative_prompt": ("STRING", {"default": ""}),
                "s_min_uncond": ("FLOAT", {"default": 0}),
                "s_churn": ("FLOAT", {"default": 0}),
                "s_tmax": ("FLOAT", {"default": 0}),
                "s_tmin": ("FLOAT", {"default": 0}),
                "s_noise": ("FLOAT", {"default": 0}),
                "refiner_checkpoint": ("STRING", {"default": ""}),
                "refiner_switch_at": ("INT", {"default": 0}),
                "disable_extra_networks": ("BOOLEAN", {"default": False}),
                "comments": ("STRING", {"default": ""}),
                "enable_hr": ("BOOLEAN", {"default": False}),
                "firstphase_width": ("INT", {"default": 0}),
                "firstphase_height": ("INT", {"default": 0}),
                "do_not_save_samples": ("BOOLEAN", {"default": False}),
                "do_not_save_grid": ("BOOLEAN", {"default": False}),
                "eta": ("FLOAT", {"default": 0}),
                "seed_resize_from_h": ("INT", {"default": 0}),
                "seed_resize_from_w": ("INT", {"default": 0})

            }
        }

    RETURN_TYPES = ("TXT2IMGRequestExtend", "DICT")
    RETURN_NAMES = ("payload", "payload_dict")
    FUNCTION = "build_payload"
    CATEGORY = "SDWebUI-API/SDWebUI-API"

    def build_payload(
        self,
        script_name: str = "",
        restore_faces: bool = False,
        tiling: bool = False,
        subseed: int = -1,
        subseed_strength: float = 0,
        styles: List[str] = [],
        sampler_index: str = "Euler a",
        script_args: List[Any] = [],
        alwayson_scripts: Dict[str, Any] = {},
        hr_scale: float = 2,
        hr_upscaler: str = "",
        hr_second_pass_steps: int = 0,
        hr_resize_x: int = 0,
        hr_resize_y: int = 0,
        hr_checkpoint_name: str = "",
        hr_sampler_name: str = "",
        hr_prompt: str = "",
        hr_negative_prompt: str = "",
        s_min_uncond: float = 0,
        s_churn: float = 0,
        s_tmax: float = 0,
        s_tmin: float = 0,
        s_noise: float = 0,
        refiner_checkpoint: str = "",
        refiner_switch_at: int = 0,
        disable_extra_networks: bool = False,
        comments: str = "",
        enable_hr: bool = False,
        firstphase_width: int = 0,
        firstphase_height: int = 0,
        do_not_save_samples: bool = False,
        do_not_save_grid: bool = False,
        eta: float = 0,
        seed_resize_from_h: int = 0,
        seed_resize_from_w: int = 0
    ):
        instance_ = TXT2IMGRequestExtend(
            script_name, restore_faces,
            tiling, subseed, subseed_strength, styles, sampler_index,
            script_args, alwayson_scripts, hr_scale, hr_upscaler,
            hr_second_pass_steps, hr_resize_x, hr_resize_y, hr_checkpoint_name, hr_sampler_name, hr_prompt, hr_negative_prompt,
             s_min_uncond, s_churn, s_tmax, s_tmin, s_noise,
             refiner_checkpoint, refiner_switch_at, disable_extra_networks, comments, enable_hr, firstphase_width,
            firstphase_height, do_not_save_samples, do_not_save_grid, eta, seed_resize_from_h, seed_resize_from_w
        )

        return instance_, dict(vars(instance_))


NODE_CLASS_MAPPINGS = {
    "SDWebUI_Request": SDWebUIRequest,
    "SDWebUI_Request_Payload": SDWebUIRequestPayload,
    "SDWebUI_Request_PayloadExtend": SDWebUIRequestPayloadExtend,
    "NovelAI_Request": NovelAIRequest,
    "Character_Prompt_Select": CharacterPromptSelect,
    "NovelAI_Request_Payload": NovelAIRequestPayload
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "SDWebUI_Request": "SDWebUIRequest",
    "SDWebUI_Request_Payload": "SDWebUIRequestPayload",
    "SDWebUI_Request_PayloadExtend": "SDWebUIRequestPayloadExtend",
    "NovelAI_Request": "NovelAIRequest",
    "Character_Prompt_Select": "CharacterPromptSelect",
    "NovelAI_Request_Payload": "NovelAIRequestPayload"

}
