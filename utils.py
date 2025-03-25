# Adapted from https://github.com/Limitex/ComfyUI-Diffusers/blob/main/utils.py

import io
import os
from typing import Optional
import torch
import requests
import numpy as np
import trimesh
from PIL import Image
from omegaconf import OmegaConf
from torchvision.transforms import ToTensor
from diffusers.pipelines.stable_diffusion.convert_from_ckpt import (
    assign_to_checkpoint,
    conv_attn_to_linear,
    create_vae_diffusers_config,
    renew_vae_attention_paths,
    renew_vae_resnet_paths,
)
from diffusers import (
    AutoencoderKL,
    DDIMScheduler,
    DDPMScheduler,
    DEISMultistepScheduler,
    DPMSolverMultistepScheduler,
    DPMSolverSinglestepScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    HeunDiscreteScheduler,
    KDPM2AncestralDiscreteScheduler,
    KDPM2DiscreteScheduler,
    UniPCMultistepScheduler,
    LCMScheduler,
    StableDiffusionXLPipeline,
)

from .mvadapter.pipelines.pipeline_mvadapter_t2mv_sdxl import MVAdapterT2MVSDXLPipeline
from .mvadapter.pipelines.pipeline_mvadapter_i2mv_sdxl import MVAdapterI2MVSDXLPipeline
from .mvadapter.pipelines.pipeline_mvadapter_i2mv_sd import MVAdapterI2MVSDPipeline
from .mvadapter.pipelines.pipeline_mvadapter_t2mv_sd import MVAdapterT2MVSDPipeline
from .mvadapter.utils import (
    get_orthogonal_camera,
    get_plucker_embeds_from_cameras_ortho,
    make_image_grid,
)
from .mvadapter.utils.render import TexturedMesh


NODE_CACHE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")

PIPELINES = {
    "StableDiffusionXLPipeline": StableDiffusionXLPipeline,
    "MVAdapterT2MVSDXLPipeline": MVAdapterT2MVSDXLPipeline,
    "MVAdapterI2MVSDXLPipeline": MVAdapterI2MVSDXLPipeline,
    "MVAdapterI2MVSDPipeline": MVAdapterI2MVSDPipeline,
    "MVAdapterT2MVSDPipeline": MVAdapterT2MVSDPipeline,
}

SCHEDULERS = {
    "DDIM": DDIMScheduler,
    "DDPM": DDPMScheduler,
    "DEISMultistep": DEISMultistepScheduler,
    "DPMSolverMultistep": DPMSolverMultistepScheduler,
    "DPMSolverSinglestep": DPMSolverSinglestepScheduler,
    "EulerAncestralDiscrete": EulerAncestralDiscreteScheduler,
    "EulerDiscrete": EulerDiscreteScheduler,
    "HeunDiscrete": HeunDiscreteScheduler,
    "KDPM2AncestralDiscrete": KDPM2AncestralDiscreteScheduler,
    "KDPM2Discrete": KDPM2DiscreteScheduler,
    "UniPCMultistep": UniPCMultistepScheduler,
    "LCM": LCMScheduler,
}

MVADAPTERS = [
    "mvadapter_t2mv_sdxl.safetensors",
    "mvadapter_i2mv_sdxl.safetensors",
    "mvadapter_i2mv_sdxl_beta.safetensors",
    "mvadapter_t2mv_sd21.safetensors",
    "mvadapter_i2mv_sd21.safetensors",
]


# Reference from : https://github.com/huggingface/diffusers/blob/main/scripts/convert_vae_pt_to_diffusers.py
def custom_convert_ldm_vae_checkpoint(checkpoint, config):
    vae_state_dict = checkpoint

    new_checkpoint = {}

    new_checkpoint["encoder.conv_in.weight"] = vae_state_dict["encoder.conv_in.weight"]
    new_checkpoint["encoder.conv_in.bias"] = vae_state_dict["encoder.conv_in.bias"]
    new_checkpoint["encoder.conv_out.weight"] = vae_state_dict[
        "encoder.conv_out.weight"
    ]
    new_checkpoint["encoder.conv_out.bias"] = vae_state_dict["encoder.conv_out.bias"]
    new_checkpoint["encoder.conv_norm_out.weight"] = vae_state_dict[
        "encoder.norm_out.weight"
    ]
    new_checkpoint["encoder.conv_norm_out.bias"] = vae_state_dict[
        "encoder.norm_out.bias"
    ]

    new_checkpoint["decoder.conv_in.weight"] = vae_state_dict["decoder.conv_in.weight"]
    new_checkpoint["decoder.conv_in.bias"] = vae_state_dict["decoder.conv_in.bias"]
    new_checkpoint["decoder.conv_out.weight"] = vae_state_dict[
        "decoder.conv_out.weight"
    ]
    new_checkpoint["decoder.conv_out.bias"] = vae_state_dict["decoder.conv_out.bias"]
    new_checkpoint["decoder.conv_norm_out.weight"] = vae_state_dict[
        "decoder.norm_out.weight"
    ]
    new_checkpoint["decoder.conv_norm_out.bias"] = vae_state_dict[
        "decoder.norm_out.bias"
    ]

    new_checkpoint["quant_conv.weight"] = vae_state_dict["quant_conv.weight"]
    new_checkpoint["quant_conv.bias"] = vae_state_dict["quant_conv.bias"]
    new_checkpoint["post_quant_conv.weight"] = vae_state_dict["post_quant_conv.weight"]
    new_checkpoint["post_quant_conv.bias"] = vae_state_dict["post_quant_conv.bias"]

    # Retrieves the keys for the encoder down blocks only
    num_down_blocks = len(
        {
            ".".join(layer.split(".")[:3])
            for layer in vae_state_dict
            if "encoder.down" in layer
        }
    )
    down_blocks = {
        layer_id: [key for key in vae_state_dict if f"down.{layer_id}" in key]
        for layer_id in range(num_down_blocks)
    }

    # Retrieves the keys for the decoder up blocks only
    num_up_blocks = len(
        {
            ".".join(layer.split(".")[:3])
            for layer in vae_state_dict
            if "decoder.up" in layer
        }
    )
    up_blocks = {
        layer_id: [key for key in vae_state_dict if f"up.{layer_id}" in key]
        for layer_id in range(num_up_blocks)
    }

    for i in range(num_down_blocks):
        resnets = [
            key
            for key in down_blocks[i]
            if f"down.{i}" in key and f"down.{i}.downsample" not in key
        ]

        if f"encoder.down.{i}.downsample.conv.weight" in vae_state_dict:
            new_checkpoint[f"encoder.down_blocks.{i}.downsamplers.0.conv.weight"] = (
                vae_state_dict.pop(f"encoder.down.{i}.downsample.conv.weight")
            )
            new_checkpoint[f"encoder.down_blocks.{i}.downsamplers.0.conv.bias"] = (
                vae_state_dict.pop(f"encoder.down.{i}.downsample.conv.bias")
            )

        paths = renew_vae_resnet_paths(resnets)
        meta_path = {"old": f"down.{i}.block", "new": f"down_blocks.{i}.resnets"}
        assign_to_checkpoint(
            paths,
            new_checkpoint,
            vae_state_dict,
            additional_replacements=[meta_path],
            config=config,
        )

    mid_resnets = [key for key in vae_state_dict if "encoder.mid.block" in key]
    num_mid_res_blocks = 2
    for i in range(1, num_mid_res_blocks + 1):
        resnets = [key for key in mid_resnets if f"encoder.mid.block_{i}" in key]

        paths = renew_vae_resnet_paths(resnets)
        meta_path = {"old": f"mid.block_{i}", "new": f"mid_block.resnets.{i - 1}"}
        assign_to_checkpoint(
            paths,
            new_checkpoint,
            vae_state_dict,
            additional_replacements=[meta_path],
            config=config,
        )

    mid_attentions = [key for key in vae_state_dict if "encoder.mid.attn" in key]
    paths = renew_vae_attention_paths(mid_attentions)
    meta_path = {"old": "mid.attn_1", "new": "mid_block.attentions.0"}
    assign_to_checkpoint(
        paths,
        new_checkpoint,
        vae_state_dict,
        additional_replacements=[meta_path],
        config=config,
    )
    conv_attn_to_linear(new_checkpoint)

    for i in range(num_up_blocks):
        block_id = num_up_blocks - 1 - i
        resnets = [
            key
            for key in up_blocks[block_id]
            if f"up.{block_id}" in key and f"up.{block_id}.upsample" not in key
        ]

        if f"decoder.up.{block_id}.upsample.conv.weight" in vae_state_dict:
            new_checkpoint[f"decoder.up_blocks.{i}.upsamplers.0.conv.weight"] = (
                vae_state_dict[f"decoder.up.{block_id}.upsample.conv.weight"]
            )
            new_checkpoint[f"decoder.up_blocks.{i}.upsamplers.0.conv.bias"] = (
                vae_state_dict[f"decoder.up.{block_id}.upsample.conv.bias"]
            )

        paths = renew_vae_resnet_paths(resnets)
        meta_path = {"old": f"up.{block_id}.block", "new": f"up_blocks.{i}.resnets"}
        assign_to_checkpoint(
            paths,
            new_checkpoint,
            vae_state_dict,
            additional_replacements=[meta_path],
            config=config,
        )

    mid_resnets = [key for key in vae_state_dict if "decoder.mid.block" in key]
    num_mid_res_blocks = 2
    for i in range(1, num_mid_res_blocks + 1):
        resnets = [key for key in mid_resnets if f"decoder.mid.block_{i}" in key]

        paths = renew_vae_resnet_paths(resnets)
        meta_path = {"old": f"mid.block_{i}", "new": f"mid_block.resnets.{i - 1}"}
        assign_to_checkpoint(
            paths,
            new_checkpoint,
            vae_state_dict,
            additional_replacements=[meta_path],
            config=config,
        )

    mid_attentions = [key for key in vae_state_dict if "decoder.mid.attn" in key]
    paths = renew_vae_attention_paths(mid_attentions)
    meta_path = {"old": "mid.attn_1", "new": "mid_block.attentions.0"}
    assign_to_checkpoint(
        paths,
        new_checkpoint,
        vae_state_dict,
        additional_replacements=[meta_path],
        config=config,
    )
    conv_attn_to_linear(new_checkpoint)
    return new_checkpoint


# Reference from : https://github.com/huggingface/diffusers/blob/main/scripts/convert_vae_pt_to_diffusers.py
def vae_pt_to_vae_diffuser(checkpoint_path: str, force_upcast: bool = True):
    try:
        config_path = os.path.join(
            NODE_CACHE_PATH, "stable-diffusion-v1-inference.yaml"
        )
        original_config = OmegaConf.load(config_path)
    except FileNotFoundError as e:
        print(f"Warning: {e}")

        r = requests.get(
            "https://raw.githubusercontent.com/CompVis/stable-diffusion/main/configs/stable-diffusion/v1-inference.yaml"
        )
        io_obj = io.BytesIO(r.content)
        original_config = OmegaConf.load(io_obj)

    image_size = 512
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if checkpoint_path.endswith("safetensors"):
        from safetensors import safe_open

        checkpoint = {}
        with safe_open(checkpoint_path, framework="pt", device="cpu") as f:
            for key in f.keys():
                checkpoint[key] = f.get_tensor(key)
    else:
        checkpoint = torch.load(checkpoint_path, map_location=device)["state_dict"]

    # Convert the VAE model.
    vae_config = create_vae_diffusers_config(original_config, image_size=image_size)
    vae_config.update({"force_upcast": force_upcast})
    converted_vae_checkpoint = custom_convert_ldm_vae_checkpoint(checkpoint, vae_config)

    vae = AutoencoderKL(**vae_config)
    vae.load_state_dict(converted_vae_checkpoint)

    return vae


def convert_images_to_tensors(images: list[Image.Image]):
    return torch.stack([np.transpose(ToTensor()(image), (1, 2, 0)) for image in images])


def convert_tensors_to_images(images: torch.tensor):
    return [
        Image.fromarray(np.clip(255.0 * image.cpu().numpy(), 0, 255).astype(np.uint8))
        for image in images
    ]


def resize_images(images: list[Image.Image], size: tuple[int, int]):
    return [image.resize(size) for image in images]


def prepare_camera_embed(num_views, size, device, azimuth_degrees=None):
    cameras = get_orthogonal_camera(
        elevation_deg=[0] * num_views,
        distance=[1.8] * num_views,
        left=-0.55,
        right=0.55,
        bottom=-0.55,
        top=0.55,
        azimuth_deg=[x - 90 for x in azimuth_degrees],
        device=device,
    )

    plucker_embeds = get_plucker_embeds_from_cameras_ortho(
        cameras.c2w, [1.1] * num_views, size
    )
    control_images = ((plucker_embeds + 1.0) / 2.0).clamp(0, 1)

    return control_images


def preprocess_image(image: Image.Image, height, width):
    image = np.array(image)
    alpha = image[..., 3] > 0
    H, W = alpha.shape
    # get the bounding box of alpha
    y, x = np.where(alpha)
    y0, y1 = max(y.min() - 1, 0), min(y.max() + 1, H)
    x0, x1 = max(x.min() - 1, 0), min(x.max() + 1, W)
    image_center = image[y0:y1, x0:x1]
    # resize the longer side to H * 0.9
    H, W, _ = image_center.shape
    if H > W:
        W = int(W * (height * 0.9) / H)
        H = int(height * 0.9)
    else:
        H = int(H * (width * 0.9) / W)
        W = int(width * 0.9)
    image_center = np.array(Image.fromarray(image_center).resize((W, H)))
    # pad to H, W
    start_h = (height - H) // 2
    start_w = (width - W) // 2
    image = np.zeros((height, width, 4), dtype=np.uint8)
    image[start_h : start_h + H, start_w : start_w + W] = image_center
    image = image.astype(np.float32) / 255.0
    image = image[:, :, :3] * image[:, :, 3:4] + (1 - image[:, :, 3:4]) * 0.5
    image = (image * 255).clip(0, 255).astype(np.uint8)
    image = Image.fromarray(image)

    return image


def load_mesh_from_trimesh(
    mesh: trimesh.Trimesh,
    rescale: bool = False,
    move_to_center: bool = False,
    scale: float = 0.5,
    flip_uv: bool = True,
    merge_vertices: bool = True,
    default_uv_size: int = 2048,
    shape_init_mesh_up: str = "+y",
    shape_init_mesh_front: str = "+x",
    front_x_to_y: bool = False,
    device: Optional[str] = None,
    return_transform: bool = False,
) -> TexturedMesh:
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.dump(concatenate=True)

    # move to center
    if move_to_center:
        centroid = mesh.vertices.mean(0)
        mesh.vertices = mesh.vertices - centroid

    # rescale
    if rescale:
        max_scale = np.abs(mesh.vertices).max()
        mesh.vertices = mesh.vertices / max_scale * scale

    dirs = ["+x", "+y", "+z", "-x", "-y", "-z"]
    dir2vec = {
        "+x": np.array([1, 0, 0]),
        "+y": np.array([0, 1, 0]),
        "+z": np.array([0, 0, 1]),
        "-x": np.array([-1, 0, 0]),
        "-y": np.array([0, -1, 0]),
        "-z": np.array([0, 0, -1]),
    }
    if shape_init_mesh_up not in dirs or shape_init_mesh_front not in dirs:
        raise ValueError(
            f"shape_init_mesh_up and shape_init_mesh_front must be one of {dirs}."
        )
    if shape_init_mesh_up[1] == shape_init_mesh_front[1]:
        raise ValueError(
            "shape_init_mesh_up and shape_init_mesh_front must be orthogonal."
        )
    z_, x_ = (
        dir2vec[shape_init_mesh_up],
        dir2vec[shape_init_mesh_front],
    )
    y_ = np.cross(z_, x_)
    std2mesh = np.stack([x_, y_, z_], axis=0).T
    mesh2std = np.linalg.inv(std2mesh)
    mesh.vertices = np.dot(mesh2std, mesh.vertices.T).T
    if front_x_to_y:
        x = mesh.vertices[:, 1].copy()
        y = -mesh.vertices[:, 0].copy()
        mesh.vertices[:, 0] = x
        mesh.vertices[:, 1] = y

    v_pos = torch.tensor(mesh.vertices, dtype=torch.float32)
    t_pos_idx = torch.tensor(mesh.faces, dtype=torch.int64)

    if hasattr(mesh, "visual") and hasattr(mesh.visual, "uv"):
        v_tex = torch.tensor(mesh.visual.uv, dtype=torch.float32)
        if flip_uv:
            v_tex[:, 1] = 1.0 - v_tex[:, 1]
        t_tex_idx = t_pos_idx.clone()
        if (
            hasattr(mesh.visual.material, "baseColorTexture")
            and mesh.visual.material.baseColorTexture
        ):
            texture = torch.tensor(
                np.array(mesh.visual.material.baseColorTexture) / 255.0,
                dtype=torch.float32,
            )[..., :3]
        else:
            texture = torch.zeros(
                (default_uv_size, default_uv_size, 3), dtype=torch.float32
            )
    else:
        v_tex = None
        t_tex_idx = None
        texture = None

    textured_mesh = TexturedMesh(
        v_pos=v_pos,
        t_pos_idx=t_pos_idx,
        v_tex=v_tex,
        t_tex_idx=t_tex_idx,
        texture=texture,
    )

    if merge_vertices:
        mesh.merge_vertices(merge_tex=True)
        textured_mesh.set_stitched_mesh(
            torch.tensor(mesh.vertices, dtype=torch.float32),
            torch.tensor(mesh.faces, dtype=torch.int64),
        )

    textured_mesh.to(device)

    if return_transform:
        return textured_mesh, np.array(centroid), max_scale / scale

    return textured_mesh