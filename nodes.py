# Adapted from https://github.com/Limitex/ComfyUI-Diffusers/blob/main/nodes.py
import copy
import os
import torch
from safetensors.torch import load_file
from torchvision import transforms
from .utils import (
    SCHEDULERS,
    PIPELINES,
    MVADAPTERS,
    load_mesh_from_trimesh,
    vae_pt_to_vae_diffuser,
    convert_images_to_tensors,
    convert_tensors_to_images,
    prepare_camera_embed,
    preprocess_image,
)
from comfy.model_management import get_torch_device
import folder_paths
import trimesh as Trimesh
from diffusers import StableDiffusionXLPipeline, AutoencoderKL, ControlNetModel
from transformers import AutoModelForImageSegmentation

from .mvadapter.pipelines.pipeline_mvadapter_t2mv_sdxl import MVAdapterT2MVSDXLPipeline
from .mvadapter.schedulers.scheduling_shift_snr import ShiftSNRScheduler
from .mvadapter.utils import get_orthogonal_camera, make_image_grid, tensor_to_image
from .mvadapter.utils.render import NVDiffRastContextWrapper, render

class DiffusersMVPipelineLoader:
    def __init__(self):
        self.hf_dir = folder_paths.get_folder_paths("diffusers")[0]
        self.dtype = torch.float16

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "ckpt_name": (
                    "STRING",
                    {"default": "stabilityai/stable-diffusion-xl-base-1.0"},
                ),
                "pipeline_name": (
                    list(PIPELINES.keys()),
                    {"default": "MVAdapterT2MVSDXLPipeline"},
                ),
            }
        }

    RETURN_TYPES = (
        "PIPELINE",
        "AUTOENCODER",
        "SCHEDULER",
    )

    FUNCTION = "create_pipeline"

    CATEGORY = "MV-Adapter"

    def create_pipeline(self, ckpt_name, pipeline_name):
        pipeline_class = PIPELINES[pipeline_name]
        pipe = pipeline_class.from_pretrained(
            pretrained_model_name_or_path=ckpt_name,
            torch_dtype=self.dtype,
            cache_dir=self.hf_dir,
        )
        return (pipe, pipe.vae, pipe.scheduler)


class LdmPipelineLoader:
    def __init__(self):
        self.hf_dir = folder_paths.get_folder_paths("diffusers")[0]
        self.dtype = torch.float16

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "ckpt_name": (folder_paths.get_filename_list("checkpoints"),),
                "pipeline_name": (
                    list(PIPELINES.keys()),
                    {"default": "MVAdapterT2MVSDXLPipeline"},
                ),
            }
        }

    RETURN_TYPES = (
        "PIPELINE",
        "AUTOENCODER",
        "SCHEDULER",
    )

    FUNCTION = "create_pipeline"

    CATEGORY = "MV-Adapter"

    def create_pipeline(self, ckpt_name, pipeline_name):
        pipeline_class = PIPELINES[pipeline_name]

        pipe = pipeline_class.from_single_file(
            pretrained_model_link_or_path=folder_paths.get_full_path(
                "checkpoints", ckpt_name
            ),
            torch_dtype=self.dtype,
            cache_dir=self.hf_dir,
        )

        return (pipe, pipe.vae, pipe.scheduler)


class DiffusersMVVaeLoader:
    def __init__(self):
        self.hf_dir = folder_paths.get_folder_paths("diffusers")[0]
        self.dtype = torch.float16

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "vae_name": (
                    "STRING",
                    {"default": "madebyollin/sdxl-vae-fp16-fix"},
                ),
            }
        }

    RETURN_TYPES = ("AUTOENCODER",)

    FUNCTION = "create_pipeline"

    CATEGORY = "MV-Adapter"

    def create_pipeline(self, vae_name):
        vae = AutoencoderKL.from_pretrained(
            pretrained_model_name_or_path=vae_name,
            torch_dtype=self.dtype,
            cache_dir=self.hf_dir,
        )

        return (vae,)


class LdmVaeLoader:
    def __init__(self):
        self.dtype = torch.float16

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "vae_name": (folder_paths.get_filename_list("vae"),),
                "upcast_fp32": ("BOOLEAN", {"default": True}),
            },
        }

    RETURN_TYPES = ("AUTOENCODER",)

    FUNCTION = "create_pipeline"

    CATEGORY = "MV-Adapter"

    def create_pipeline(self, vae_name, upcast_fp32):
        vae = vae_pt_to_vae_diffuser(
            folder_paths.get_full_path("vae", vae_name), force_upcast=upcast_fp32
        ).to(self.dtype)

        return (vae,)


class DiffusersMVSchedulerLoader:
    def __init__(self):
        self.hf_dir = folder_paths.get_folder_paths("diffusers")[0]
        self.dtype = torch.float16

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "pipeline": ("PIPELINE",),
                "scheduler_name": (list(SCHEDULERS.keys()),),
                "shift_snr": ("BOOLEAN", {"default": True}),
                "shift_mode": (
                    ["default", "interpolated"],
                    {"default": "interpolated"},
                ),
                "shift_scale": (
                    "FLOAT",
                    {"default": 8.0, "min": 0.0, "max": 50.0, "step": 1.0},
                ),
            }
        }

    RETURN_TYPES = ("SCHEDULER",)

    FUNCTION = "load_scheduler"

    CATEGORY = "MV-Adapter"

    def load_scheduler(
        self, pipeline, scheduler_name, shift_snr, shift_mode, shift_scale
    ):
        scheduler = SCHEDULERS[scheduler_name].from_config(
            pipeline.scheduler.config, torch_dtype=self.dtype
        )
        if shift_snr:
            scheduler = ShiftSNRScheduler.from_scheduler(
                scheduler,
                shift_mode=shift_mode,
                shift_scale=shift_scale,
                scheduler_class=scheduler.__class__,
            )
        return (scheduler,)


class LoraModelLoader:
    def __init__(self):
        self.loaded_lora = None

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "pipeline": ("PIPELINE",),
                "lora_name": (folder_paths.get_filename_list("loras"),),
                "strength_model": (
                    "FLOAT",
                    {"default": 1.0, "min": -100.0, "max": 100.0, "step": 0.01},
                ),
            }
        }

    RETURN_TYPES = ("PIPELINE",)
    FUNCTION = "load_lora"
    CATEGORY = "MV-Adapter"

    def load_lora(self, pipeline, lora_name, strength_model):
        if strength_model == 0:
            return (pipeline,)

        lora_path = folder_paths.get_full_path("loras", lora_name)
        lora_dir = os.path.dirname(lora_path)
        lora_name = os.path.basename(lora_path)
        lora = None
        if self.loaded_lora is not None:
            if self.loaded_lora[0] == lora_path:
                lora = self.loaded_lora[1]
            else:
                temp = self.loaded_lora
                pipeline.delete_adapters(temp[1])
                self.loaded_lora = None

        if lora is None:
            adapter_name = lora_name.rsplit(".", 1)[0]
            pipeline.load_lora_weights(
                lora_dir, weight_name=lora_name, adapter_name=adapter_name
            )
            pipeline.set_adapters(adapter_name, strength_model)
            self.loaded_lora = (lora_path, adapter_name)
            lora = adapter_name

        return (pipeline,)


class ControlNetModelLoader:
    def __init__(self):
        self.loaded_controlnet = None
        self.dtype = torch.float16
        self.torch_device = get_torch_device()
        self.hf_dir = folder_paths.get_folder_paths("diffusers")[0]

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "pipeline": ("PIPELINE",),
                "controlnet_name": (
                    "STRING",
                    {"default": "xinsir/controlnet-scribble-sdxl-1.0"},
                ),
            }
        }

    RETURN_TYPES = ("PIPELINE",)
    FUNCTION = "load_controlnet"
    CATEGORY = "MV-Adapter"

    def load_controlnet(self, pipeline, controlnet_name):
        controlnet = None
        if self.loaded_controlnet is not None:
            if self.loaded_controlnet == controlnet_name:
                controlnet = self.loaded_controlnet
            else:
                del pipeline.controlnet
                self.loaded_controlnet = None

        if controlnet is None:
            controlnet = ControlNetModel.from_pretrained(
                controlnet_name, cache_dir=self.hf_dir, torch_dtype=self.dtype
            )
            pipeline.controlnet = controlnet
            pipeline.controlnet.to(device=self.torch_device, dtype=self.dtype)

            self.loaded_controlnet = controlnet_name
            controlnet = controlnet_name

        return (pipeline,)


class DiffusersMVModelMakeup:
    def __init__(self):
        self.hf_dir = folder_paths.get_folder_paths("diffusers")[0]
        self.torch_device = get_torch_device()
        self.dtype = torch.float16

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "pipeline": ("PIPELINE",),
                "scheduler": ("SCHEDULER",),
                "autoencoder": ("AUTOENCODER",),
                "load_mvadapter": ("BOOLEAN", {"default": True}),
                "adapter_path": ("STRING", {"default": "huanngzh/mv-adapter"}),
                "adapter_name": (
                    MVADAPTERS,
                    {"default": "mvadapter_t2mv_sdxl.safetensors"},
                ),
                "num_views": ("INT", {"default": 6, "min": 1, "max": 12}),
            },
            "optional": {
                "enable_vae_slicing": ("BOOLEAN", {"default": True}),
                "enable_vae_tiling": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("PIPELINE",)

    FUNCTION = "makeup_pipeline"

    CATEGORY = "MV-Adapter"

    def makeup_pipeline(
        self,
        pipeline,
        scheduler,
        autoencoder,
        load_mvadapter,
        adapter_path,
        adapter_name,
        num_views,
        enable_vae_slicing=True,
        enable_vae_tiling=False,
    ):
        pipeline.vae = autoencoder
        pipeline.scheduler = scheduler

        if load_mvadapter:
            pipeline.init_custom_adapter(num_views=num_views)
            pipeline.load_custom_adapter(
                adapter_path, weight_name=adapter_name, cache_dir=self.hf_dir
            )
            pipeline.cond_encoder.to(device=self.torch_device, dtype=self.dtype)

        pipeline = pipeline.to(self.torch_device, self.dtype)

        if enable_vae_slicing:
            pipeline.enable_vae_slicing()
        if enable_vae_tiling:
            pipeline.enable_vae_tiling()

        return (pipeline,)


class DiffusersSampler:
    def __init__(self):
        self.torch_device = get_torch_device()

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "pipeline": ("PIPELINE",),
                "prompt": (
                    "STRING",
                    {"multiline": True, "default": "a photo of a cat"},
                ),
                "negative_prompt": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "watermark, ugly, deformed, noisy, blurry, low contrast",
                    },
                ),
                "width": ("INT", {"default": 768, "min": 1, "max": 2048, "step": 1}),
                "height": ("INT", {"default": 768, "min": 1, "max": 2048, "step": 1}),
                "steps": ("INT", {"default": 50, "min": 1, "max": 2000}),
                "cfg": (
                    "FLOAT",
                    {
                        "default": 7.0,
                        "min": 0.0,
                        "max": 100.0,
                        "step": 0.1,
                        "round": 0.01,
                    },
                ),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
            }
        }

    RETURN_TYPES = ("IMAGE",)

    FUNCTION = "sample"

    CATEGORY = "MV-Adapter"

    def sample(
        self,
        pipeline,
        prompt,
        negative_prompt,
        height,
        width,
        steps,
        cfg,
        seed,
    ):
        images = pipeline(
            prompt=prompt,
            height=height,
            width=width,
            num_inference_steps=steps,
            guidance_scale=cfg,
            negative_prompt=negative_prompt,
            generator=torch.Generator(self.torch_device).manual_seed(seed),
        ).images
        return (convert_images_to_tensors(images),)


class DiffusersMVSampler:
    def __init__(self):
        self.torch_device = get_torch_device()

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "pipeline": ("PIPELINE",),
                "num_views": ("INT", {"default": 6, "min": 1, "max": 12}),
                "prompt": (
                    "STRING",
                    {"multiline": True, "default": "an astronaut riding a horse"},
                ),
                "negative_prompt": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "watermark, ugly, deformed, noisy, blurry, low contrast",
                    },
                ),
                "width": ("INT", {"default": 768, "min": 1, "max": 2048, "step": 1}),
                "height": ("INT", {"default": 768, "min": 1, "max": 2048, "step": 1}),
                "steps": ("INT", {"default": 50, "min": 1, "max": 2000}),
                "cfg": (
                    "FLOAT",
                    {
                        "default": 7.0,
                        "min": 0.0,
                        "max": 100.0,
                        "step": 0.1,
                        "round": 0.01,
                    },
                ),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
            },
            "optional": {
                "reference_image": ("IMAGE",),
                "controlnet_image": ("IMAGE",),
                "controlnet_conditioning_scale": ("FLOAT", {"default": 1.0}),
                "azimuth_degrees": ("LIST", {"default": [0, 45, 90, 180, 270, 315]}),
            },
        }

    RETURN_TYPES = ("IMAGE",)

    FUNCTION = "sample"

    CATEGORY = "MV-Adapter"

    def sample(
        self,
        pipeline,
        num_views,
        prompt,
        negative_prompt,
        height,
        width,
        steps,
        cfg,
        seed,
        reference_image=None,
        controlnet_image=None,
        controlnet_conditioning_scale=1.0,
        azimuth_degrees=[0, 45, 90, 180, 270, 315],
    ):
        num_views = len(azimuth_degrees)
        control_images = prepare_camera_embed(
            num_views, width, self.torch_device, azimuth_degrees
        )

        pipe_kwargs = {}
        if reference_image is not None:
            pipe_kwargs.update(
                {
                    "reference_image": convert_tensors_to_images(reference_image)[0],
                    "reference_conditioning_scale": 1.0,
                }
            )
        if controlnet_image is not None:
            controlnet_image = convert_tensors_to_images(controlnet_image)
            pipe_kwargs.update(
                {
                    "controlnet_image": controlnet_image,
                    "controlnet_conditioning_scale": controlnet_conditioning_scale,
                }
            )

        images = pipeline(
            prompt=prompt,
            height=height,
            width=width,
            num_inference_steps=steps,
            guidance_scale=cfg,
            num_images_per_prompt=num_views,
            control_image=control_images,
            control_conditioning_scale=1.0,
            negative_prompt=negative_prompt,
            generator=torch.Generator(self.torch_device).manual_seed(seed),
            cross_attention_kwargs={"num_views": num_views},
            **pipe_kwargs,
        ).images
        return (convert_images_to_tensors(images),)


class BiRefNet:
    def __init__(self):
        self.hf_dir = folder_paths.get_folder_paths("diffusers")[0]
        self.torch_device = get_torch_device()
        self.dtype = torch.float32

    RETURN_TYPES = ("FUNCTION",)

    FUNCTION = "load_model_fn"

    CATEGORY = "MV-Adapter"

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {"ckpt_name": ("STRING", {"default": "ZhengPeng7/BiRefNet"})}
        }

    def remove_bg(self, image, net, transform, device):
        image_size = image.size
        input_images = transform(image).unsqueeze(0).to(device)
        with torch.no_grad():
            preds = net(input_images)[-1].sigmoid().cpu()
        pred = preds[0].squeeze()
        pred_pil = transforms.ToPILImage()(pred)
        mask = pred_pil.resize(image_size)
        image.putalpha(mask)
        return image

    def load_model_fn(self, ckpt_name):
        model = AutoModelForImageSegmentation.from_pretrained(
            ckpt_name, trust_remote_code=True, cache_dir=self.hf_dir
        ).to(self.torch_device, self.dtype)

        transform_image = transforms.Compose(
            [
                transforms.Resize((1024, 1024)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

        remove_bg_fn = lambda x: self.remove_bg(
            x, model, transform_image, self.torch_device
        )
        return (remove_bg_fn,)


class ImagePreprocessor:
    def __init__(self):
        self.torch_device = get_torch_device()

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "remove_bg_fn": ("FUNCTION",),
                "image": ("IMAGE",),
                "height": ("INT", {"default": 768, "min": 1, "max": 2048, "step": 1}),
                "width": ("INT", {"default": 768, "min": 1, "max": 2048, "step": 1}),
            }
        }

    RETURN_TYPES = ("IMAGE",)

    FUNCTION = "process"

    def process(self, remove_bg_fn, image, height, width):
        images = convert_tensors_to_images(image)
        images = [
            preprocess_image(remove_bg_fn(img.convert("RGB")), height, width)
            for img in images
        ]

        return (convert_images_to_tensors(images),)


class ControlImagePreprocessor:
    def __init__(self):
        self.torch_device = get_torch_device()

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "front_view": ("IMAGE",),
                "front_right_view": ("IMAGE",),
                "right_view": ("IMAGE",),
                "back_view": ("IMAGE",),
                "left_view": ("IMAGE",),
                "front_left_view": ("IMAGE",),
                "width": ("INT", {"default": 768, "min": 1, "max": 2048, "step": 1}),
                "height": ("INT", {"default": 768, "min": 1, "max": 2048, "step": 1}),
            }
        }

    RETURN_TYPES = ("IMAGE",)

    FUNCTION = "process"

    def process(
        self,
        front_view,
        front_right_view,
        right_view,
        back_view,
        left_view,
        front_left_view,
        width,
        height,
    ):
        images = torch.cat(
            [
                front_view,
                front_right_view,
                right_view,
                back_view,
                left_view,
                front_left_view,
            ],
            dim=0,
        )
        images = convert_tensors_to_images(images)
        images = [img.resize((width, height)).convert("RGB") for img in images]
        return (convert_images_to_tensors(images),)


class ViewSelector:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "front_view": ("BOOLEAN", {"default": True}),
                "front_right_view": ("BOOLEAN", {"default": True}),
                "right_view": ("BOOLEAN", {"default": True}),
                "back_view": ("BOOLEAN", {"default": True}),
                "left_view": ("BOOLEAN", {"default": True}),
                "front_left_view": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("LIST",)
    FUNCTION = "process"
    CATEGORY = "MV-Adapter"

    def process(
        self,
        front_view,
        front_right_view,
        right_view,
        back_view,
        left_view,
        front_left_view,
    ):
        azimuth_deg = []
        if front_view:
            azimuth_deg.append(0)
        if front_right_view:
            azimuth_deg.append(45)
        if right_view:
            azimuth_deg.append(90)
        if back_view:
            azimuth_deg.append(180)
        if left_view:
            azimuth_deg.append(270)
        if front_left_view:
            azimuth_deg.append(315)

        return (azimuth_deg,)



class MVAdapterLoadMesh:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "glb_path": ("STRING", {"default": "", "tooltip": "The glb path with mesh to load."}), 
            }
        }
    RETURN_TYPES = ("TRIMESH",)
    RETURN_NAMES = ("trimesh",)
    OUTPUT_TOOLTIPS = ("The glb model with mesh to texturize.",)
    
    FUNCTION = "load"
    CATEGORY = "MV-Adapter"
    DESCRIPTION = "Loads a glb model from the given path."

    def load(self, glb_path):        
        trimesh = Trimesh.load(glb_path, force="mesh")        
        return (trimesh,)


class MVAdapterCameraConfig:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "camera_azimuths": ("STRING", {"default": "-90, 0, 90, 180, 90, 90", "multiline": False}),
                "camera_elevations": ("STRING", {"default": "0, 0, 0, 0, 90, -90", "multiline": False}),
                "camera_distance": ("FLOAT", {"default": 1.8, "min": 0.1, "max": 10.0, "step": 0.001}),
                "ortho_scale": ("FLOAT", {"default": 1.1, "min": 0.1, "max": 2, "step": 0.001}),
            },
        }

    RETURN_TYPES = ("MVADAPTER_CAMERA",)
    RETURN_NAMES = ("camera_config",)
    FUNCTION = "process"
    CATEGORY = "MV-Adapter"

    def process(self, camera_azimuths, camera_elevations, camera_distance, ortho_scale):
        angles_list = list(map(float, camera_azimuths.replace(" ", "").split(',')))
        elevations_list = list(map(float, camera_elevations.replace(" ", "").split(',')))
        if len(angles_list) != len(elevations_list):
            raise ValueError("Count of azimuth and elevation angles do not match")        
        dist_list = [camera_distance] * len(angles_list)        

        camera_config = {
            "camera_azims": angles_list,
            "camera_elevs": elevations_list,
            "camera_dist": camera_distance,
            "camera_dists": dist_list,            
            "ortho_scale": ortho_scale,
            "left": -0.5*ortho_scale,
            "right": 0.5*ortho_scale,
            "bottom": -0.5*ortho_scale,
            "top": 0.5*ortho_scale,
            }
        
        return (camera_config,)


class MVAdapterRenderMultiView:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "trimesh": ("TRIMESH",),
                "render_size": ("INT", {"default": 768, "min": 64, "max": 2048, "step": 16}),
                "texture_size": ("INT", {"default": 768, "min": 64, "max": 2048, "step": 16}),
            },
            "optional": {
                "camera_config": ("MVADAPTER_CAMERA",),
            }
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "MASK",)
    RETURN_NAMES = ("normal_maps", "position_maps", "masks")
    FUNCTION = "process"
    CATEGORY = "MV-Adapter"

    def __init__(self):
        self.loaded_controlnet = None
        self.dtype = torch.float16
        self.torch_device = get_torch_device()

    def process(self, trimesh, render_size, texture_size, camera_config=None):
        if camera_config == None:
            camera_config = {
                "camera_azims": [-90, 0, 90, 180, 90, 90],
                "camera_elevs": [0, 0, 0, 0, 89.99, -89.99],
                "camera_dist": 1.8,
                "camera_dists": [1.8] * 6,            
                "ortho_scale": 1.1,
                "left": -0.55,
                "right": 0.55,
                "bottom": -0.55,
                "top": 0.55,
            }

        cameras = get_orthogonal_camera(
            elevation_deg = camera_config['camera_elevs'],
            azimuth_deg = camera_config['camera_azims'],
            distance = camera_config['camera_dists'],
            left=camera_config['left'],
            right=camera_config['right'],
            bottom=camera_config['bottom'],
            top=camera_config['top'],
            device=self.torch_device,
        )
        ctx = NVDiffRastContextWrapper(device=self.torch_device)

        mesh = load_mesh_from_trimesh(trimesh.copy(), rescale=True, device=self.torch_device)

        render_out = render(
            ctx,
            mesh,
            cameras,
            height=render_size,
            width=render_size,
            render_attr=False,
            normal_background=0.0,
        )

        return (
            (render_out.normal / 2 + 0.5).clamp(0, 1).cpu().float(), 
            (render_out.pos + 0.5).clamp(0, 1).cpu().float(), 
            render_out.mask.squeeze(-1).cpu().float(),)

        return ([],[],[])


NODE_CLASS_MAPPINGS = {
    "LdmPipelineLoader": LdmPipelineLoader,
    "LdmVaeLoader": LdmVaeLoader,
    "DiffusersMVPipelineLoader": DiffusersMVPipelineLoader,
    "DiffusersMVVaeLoader": DiffusersMVVaeLoader,
    "DiffusersMVSchedulerLoader": DiffusersMVSchedulerLoader,
    "DiffusersMVModelMakeup": DiffusersMVModelMakeup,
    "LoraModelLoader": LoraModelLoader,
    "DiffusersMVSampler": DiffusersMVSampler,
    "BiRefNet": BiRefNet,
    "ImagePreprocessor": ImagePreprocessor,
    "ControlNetModelLoader": ControlNetModelLoader,
    "ControlImagePreprocessor": ControlImagePreprocessor,
    "ViewSelector": ViewSelector,
    "MVAdapterLoadMesh": MVAdapterLoadMesh,
    "MVAdapterCameraConfig": MVAdapterCameraConfig,
    "MVAdapterRenderMultiView": MVAdapterRenderMultiView,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LdmPipelineLoader": "LDM Pipeline Loader",
    "LdmVaeLoader": "LDM Vae Loader",
    "DiffusersMVPipelineLoader": "Diffusers Pipeline Loader",
    "DiffusersMVVaeLoader": "Diffusers Vae Loader",
    "DiffusersMVSchedulerLoader": "Diffusers Scheduler Loader",
    "DiffusersMVModelMakeup": "Diffusers Model Makeup",
    "LoraModelLoader": "Lora Model Loader",
    "DiffusersMVSampler": "Diffusers MV Sampler",
    "BiRefNet": "BiRefNet",
    "ImagePreprocessor": "Image Preprocessor",
    "ControlNetModelLoader": "ControlNet Model Loader",
    "ControlImagePreprocessor": "Control Image Preprocessor",
    "ViewSelector": "View Selector",
    "MVAdapterLoadMesh": "MV-Adapter Load Mesh",
    "MVAdapterCameraConfig": "MV-Adapter Camera Config",
    "MVAdapterRenderMultiView": "MV-Adapter Render Multi-View",
}
