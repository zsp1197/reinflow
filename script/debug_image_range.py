import os
import json
import numpy as np
import torch
import robomimic.utils.env_utils as EnvUtils
import robomimic.utils.obs_utils as ObsUtils
from omegaconf import OmegaConf
from env.gym_utils.wrapper.robomimic_image import RobomimicImageWrapper

os.environ["MUJOCO_GL"] = "egl"


def check_image_range():
    cfg_path = "cfg/robomimic/finetune/can/ft_ppo_reflow_mlp_img.yaml"
    cfg = OmegaConf.load(cfg_path)
    shape_meta = cfg["shape_meta"]

    wrappers = cfg.env.wrappers
    obs_modality_dict = {
        "low_dim": wrappers.robomimic_image.low_dim_keys,
        "rgb": wrappers.robomimic_image.image_keys,
    }
    ObsUtils.initialize_obs_modality_mapping_from_dict(obs_modality_dict)

    with open(cfg.robomimic_env_cfg_path, "r") as f:
        env_meta = json.load(f)

    env = EnvUtils.create_env_from_metadata(
        env_meta=env_meta,
        render=False,
        render_offscreen=True,
        use_image_obs=True,
    )

    raw_obs = env.reset()
    img_key = cfg.env.wrappers.robomimic_image.image_keys[0]
    raw_img = raw_obs[img_key]
    print(
        f"Raw image (normalized by robomimic?) range: [{raw_img.min()}, {raw_img.max()}], dtype: {raw_img.dtype}"
    )

    wrapper = RobomimicImageWrapper(
        env=env,
        shape_meta=shape_meta,
        image_keys=cfg.env.wrappers.robomimic_image.image_keys,
        low_dim_keys=cfg.env.wrappers.robomimic_image.low_dim_keys,
    )

    obs = wrapper.reset()
    wrapped_img = obs["rgb"]
    print(
        f"Wrapped image range: [{wrapped_img.min()}, {wrapped_img.max()}], dtype: {wrapped_img.dtype}"
    )

    # Check what VitEncoder does
    from model.common.vit import VitEncoder, VitEncoderConfig

    obs_shape = shape_meta.obs.rgb.shape
    enc = VitEncoder(
        obs_shape,
        VitEncoderConfig(),
        num_channel=obs_shape[0],
        img_h=obs_shape[1],
        img_w=obs_shape[2],
    )

    # Pass wrapped_img through encoder forward (simulating how it's used in training)
    # The agent passes a batch, so we add a batch dim
    input_tensor = torch.from_numpy(wrapped_img).unsqueeze(0).float()
    print(
        f"Input to encoder min/max: {input_tensor.min().item()}, {input_tensor.max().item()}"
    )

    # Inside VitEncoder.forward: obs = obs / 255.0 - 0.5
    normalized_input = input_tensor / 255.0 - 0.5
    print(
        f"Normalized input (inside encoder) min/max: {normalized_input.min().item()}, {normalized_input.max().item()}"
    )


if __name__ == "__main__":
    check_image_range()
