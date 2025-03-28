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
from .mvadapter.utils.geometry import get_plucker_embeds_from_cameras_ortho

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
                "control_image": ("IMAGE",),
                "control_conditioning_scale": ("FLOAT", {"default": 1.0}),
                "camera_config": ("MVADAPTER_CAMERA",),
            },
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
        reference_image=None,
        control_image=None,
        control_conditioning_scale=1.0,
        camera_config=None,
    ):
        if camera_config is None:
            camera_config = MVAdapterCameraConfig.get_default_camera_config()
        
        assert 'camera_elevs' in camera_config
        assert 'camera_azims' in camera_config
        
        num_views = len(camera_config['camera_azims'])
        assert num_views == len(camera_config['camera_elevs'])

        if control_image is not None:
            if isinstance(control_image, list) or isinstance(control_image, tuple):
                pos_maps = control_image[0]
                nrm_maps = control_image[1]
            else:
                pos_maps = control_image[:num_views]
                nrm_maps = control_image[num_views:]
            control_image = (
                torch.cat(
                    [
                        pos_maps,
                        nrm_maps
                    ],
                    dim=-1,
                )
                .permute(0, 3, 1, 2)
                .to(self.torch_device)
            )
        else:
            cameras = get_orthogonal_camera(
                elevation_deg=camera_config['camera_elevs'],
                azimuth_deg=[x-90 for x in camera_config['camera_azims']],
                distance=[camera_config.get('camera_dist', 1.8)] * num_views,
                left=camera_config.get('left', -0.55),
                right=camera_config.get('right', 0.55),
                bottom=camera_config.get('bottom', -0.55),
                top=camera_config.get('top', 0.55),
                device=self.torch_device,
            )

            plucker_embeds = get_plucker_embeds_from_cameras_ortho(
                cameras.c2w, [camera_config.get('ortho_scale', 1.1)] * num_views, width
            )
            control_image = ((plucker_embeds + 1.0) / 2.0).clamp(0, 1)
            print(control_image.shape)

        pipe_kwargs = {}
        if reference_image is not None:
            pipe_kwargs.update(
                {
                    "reference_image": convert_tensors_to_images(reference_image)[0],
                    "reference_conditioning_scale": 1.0,
                }
            )
        if control_image is not None:
            pipe_kwargs.update(
                {
                    "control_image": control_image,
                    "control_conditioning_scale": control_conditioning_scale,
                }
            )

        images = pipeline(
            prompt=prompt,
            negative_prompt=negative_prompt,
            height=height,
            width=width,
            num_inference_steps=steps,
            guidance_scale=cfg,
            num_images_per_prompt=num_views,
            generator=torch.Generator(self.torch_device).manual_seed(seed),
            cross_attention_kwargs={"scale": 1.0, "num_views": num_views},
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
    
class MVAdapterNormalizeMesh:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "trimesh": ("TRIMESH",), 
                "size": ("FLOAT", {"default": 1.0, "min": 0.0}),
                "flip_x": ("BOOLEAN", {"default": False}),
                "flip_y": ("BOOLEAN", {"default": False}),
                "flip_z": ("BOOLEAN", {"default": False}),
            }
        }
    RETURN_TYPES = ("TRIMESH",)
    RETURN_NAMES = ("trimesh",)
    
    FUNCTION = "process"
    CATEGORY = "MV-Adapter"
    DESCRIPTION = "Resize the mesh based on its most significant axis."

    def process(self, trimesh, size=1.0, flip_x=False, flip_y=False, flip_z=False):
        min_bound = trimesh.bounds.min(axis=0)
        max_bound = trimesh.bounds.max(axis=0)
        extents = max_bound - min_bound

        current_longest_length = extents.max()

        if current_longest_length <= 1e-6:
            scale_factor = 1.0
        else:
            scale_factor = size / current_longest_length

        scales = [scale_factor] * 3
        if flip_x:
            scales[0] = -scale_factor
        if flip_y:
            scales[1] = -scale_factor
        if flip_z:
            scales[2] = -scale_factor
        
        transformed_mesh = trimesh.copy()
        transformed_mesh.apply_scale(scales)

        return (transformed_mesh,)

class MVAdapterCameraConfig:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "camera_azimuths": ("STRING", {"default": "0, 90, 180, 270, 180, 180", "multiline": False}),
                "camera_elevations": ("STRING", {"default": "0, 0, 0, 0, 90, -90", "multiline": False}),
                "camera_distance": ("FLOAT", {"default": 1.8, "min": 0.1, "max": 10.0, "step": 0.001}),
                "ortho_scale": ("FLOAT", {"default": 1.1, "min": 0.1, "max": 2, "step": 0.001}),
            },
        }

    RETURN_TYPES = ("MVADAPTER_CAMERA",)
    RETURN_NAMES = ("camera_config",)
    FUNCTION = "process"
    CATEGORY = "MV-Adapter"

    @classmethod
    def get_default_camera_config(cls):
        return {
            "camera_azims": [0, 90, 180, 270, 180, 180],
            "camera_elevs": [0, 0, 0, 0, 89.99, -89.99],
            "camera_dist": 1.8,         
            "ortho_scale": 1.1,
            "left": -0.55,
            "right": 0.55,
            "bottom": -0.55,
            "top": 0.55,
        }

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
    RETURN_NAMES = ("position_maps", "normal_maps", "masks")
    FUNCTION = "process"
    CATEGORY = "MV-Adapter"

    def __init__(self):
        self.loaded_controlnet = None
        self.dtype = torch.float16
        self.torch_device = get_torch_device()

    def process(self, trimesh, render_size, texture_size, camera_config=None):
        if camera_config == None:
            camera_config = MVAdapterCameraConfig.get_default_camera_config()

        num_views = len(camera_config['camera_azims'])

        cameras = get_orthogonal_camera(
            elevation_deg = camera_config['camera_elevs'],
            azimuth_deg = [x - 90 for x in camera_config['camera_azims']],
            distance = [camera_config['camera_dist']] * num_views,
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
            (render_out.pos + 0.5).clamp(0, 1).cpu().float(), 
            (render_out.normal / 2 + 0.5).clamp(0, 1).cpu().float(), 
            render_out.mask.squeeze(-1).cpu().float(),
        )

class FillBackgroundWithColor:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),  # Input RGB or RGBA image tensor
                "red": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "green": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "blue": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "invert_alpha": ("BOOLEAN", {"default": False}),                 
                "output_alpha": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "mask": ("MASK",),  # Optional mask input
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    CATEGORY = "MV-Adapter"
    FUNCTION = "process"

    def process(self, image, red, green, blue, invert_alpha=False, output_alpha=False, mask=None):
        background_color = torch.tensor([red, green, blue], dtype=image.dtype, device=image.device).view(1, 1, 1, 3)
        
        assert (mask is None) or (image.shape[:3] == mask.shape[:3])

        output_images = []

        # for img in image: # Iterate through batch if batch_size > 1
        for i in range(image.shape[0]):            
            img = image[i]
            msk = mask[i] if mask is not None else None
            
            rgba_image = img.clone() # Clone to avoid modifying the original image
            original_channels = rgba_image.shape[-1]
            
            alpha_channel = None # Initialize alpha_channel

            if msk is not None:
                # Use provided mask as alpha
                mask_expanded = msk.float() # Ensure mask is float
                if mask_expanded.ndim == 2: # (H, W) -> (H, W, 1)
                    mask_expanded = mask_expanded.unsqueeze(-1)

                alpha_channel = mask_expanded

                if original_channels == 3: # RGB image with mask, convert to RGBA
                    rgba_image = torch.cat([rgba_image, alpha_channel], dim=-1)
                elif original_channels == 4: # RGBA image with mask, replace alpha
                    rgba_image[:, :, 3:4] = alpha_channel
                else:
                    raise ValueError(f"Input image should have 3 (RGB) or 4 (RGBA) channels, but got {original_channels}")

            elif original_channels == 4:
                # Use existing alpha channel of RGBA image
                alpha_channel = rgba_image[:, :, 3:4]
            elif original_channels == 3:
                # RGB image without mask, treat as opaque (alpha = 1)
                alpha_channel = torch.ones_like(rgba_image[:,:,:1]) # Create an alpha channel of ones
                rgba_image = torch.cat([rgba_image, alpha_channel], dim=-1) # Convert RGB to RGBA
            else:
                raise ValueError(f"Input image should have 3 (RGB) or 4 (RGBA) channels, but got {original_channels}")

            if alpha_channel is None: # Safety check in case alpha_channel was not assigned.
                raise Exception("Alpha channel was not properly assigned.")


            rgb_channels = rgba_image[:, :, :3] # Extract RGB channels (now guaranteed to be RGB part of RGBA)
            # alpha_channel is already extracted above

            if invert_alpha:
                alpha_channel = 1 - alpha_channel
                
            # Blend the background color with the original RGB based on alpha
            blended_rgb = (rgb_channels * alpha_channel) + (background_color * (1 - alpha_channel))

            rgba_image[:, :, :3] = blended_rgb # Replace RGB channels with blended ones

            if not output_alpha:
                rgba_image = rgba_image[:, :, :3] # Remove alpha channel if not required

            output_images.append(rgba_image)

        output_image_batch = torch.stack(output_images)
        return (output_image_batch,)


class InvertChannelsOfImages:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),  # Input RGB or RGBA image tensor
                "red": ("STRING", {"default": "no"}),
                "green": ("STRING", {"default": "no"}),
                "blue": ("STRING", {"default": "no"}),
                "alpha": ("STRING", {"default": "no"}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    CATEGORY = "MV-Adapter"
    FUNCTION = "process"

    def process(self, image, red, green, blue, alpha):
        output_images = []
        image_count = image.shape[0]

        def get_flip_flag(x):
            return (x.lower() in ['1', 'true', 'true', 'yes', 'on', 'flip', 'invert'])
        
        def parse_list(chn):
            l = list(map(get_flip_flag, chn.replace(" ", "").split(',')))
            l += [l[-1]] * max(0, image_count - len(l))
            return l
        
        red_list = parse_list(red)
        green_list = parse_list(green)
        blue_list = parse_list(blue)
        alpha_list = parse_list(alpha)

        # for img in image: # Iterate through batch if batch_size > 1
        for i in range(image_count):            
            img = image[i]
            
            rgba_image = img.clone() # Clone to avoid modifying the original image

            if red_list[i]:                
                rgba_image[:,:,0] = 1 - rgba_image[:,:,0]
            if green_list[i]:                
                rgba_image[:,:,1] = 1 - rgba_image[:,:,1]
            if blue_list[i]:                
                rgba_image[:,:,2] = 1 - rgba_image[:,:,2]
            if alpha_list[i] and rgba_image.shape[-1] == 4:                
                rgba_image[:,:,3] = 1 - rgba_image[:,:,3]

            output_images.append(rgba_image)

        output_image_batch = torch.stack(output_images)
        return (output_image_batch,)

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
    #"ViewSelector": ViewSelector,
    "MVAdapterNormalizeMesh": MVAdapterNormalizeMesh,
    "MVAdapterLoadMesh": MVAdapterLoadMesh,
    "MVAdapterCameraConfig": MVAdapterCameraConfig,
    "MVAdapterRenderMultiView": MVAdapterRenderMultiView,
    "FillBackgroundWithColor": FillBackgroundWithColor,
    "InvertChannelsOfImages": InvertChannelsOfImages,
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
    #"ViewSelector": "View Selector",
    "MVAdapterNormalizeMesh": "MV-Adapter Normalize Mesh",
    "MVAdapterLoadMesh": "MV-Adapter Load Mesh",
    "MVAdapterCameraConfig": "MV-Adapter Camera Config",
    "MVAdapterRenderMultiView": "MV-Adapter Render Multi-View",
    "FillBackgroundWithColor": "Fill Background With Color",
    "InvertChannelsOfImages": "Invert Channels of Images",
}
