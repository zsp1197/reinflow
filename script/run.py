# MIT License
#
# Copyright (c) 2024 Intelligent Robot Motion Lab
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import os
import sys
import logging
import math
import gc

# Automatically add project root to sys.path
sys.path.append(os.getcwd())

import hydra
from omegaconf import OmegaConf
import gdown
from download_url import (
    get_dataset_download_url,
    get_normalization_download_url,
    get_checkpoint_download_url,
)

# Set defaults before hydra.main
os.environ.setdefault("REINFLOW_LOG_DIR", os.path.abspath("log"))
os.environ.setdefault("REINFLOW_DATA_DIR", os.path.abspath("data"))
os.environ.setdefault("REINFLOW_WANDB_ENTITY", "")
os.environ.setdefault("DPPO_LOG_DIR", os.path.abspath("log"))
os.environ.setdefault("DPPO_DATA_DIR", os.path.abspath("data"))
os.environ.setdefault("DPPO_WANDB_ENTITY", "")

# If no W&B entity is provided, default to offline mode
if not os.environ.get("REINFLOW_WANDB_ENTITY") and not os.environ.get(
    "DPPO_WANDB_ENTITY"
):
    os.environ.setdefault("WANDB_MODE", "offline")

os.environ["D4RL_SUPPRESS_IMPORT_ERROR"] = "1"

# register kitchen tasks
import d4rl.gym_mujoco

gc.collect()

OmegaConf.register_new_resolver("eval", eval, replace=True)
OmegaConf.register_new_resolver("round_up", math.ceil)
OmegaConf.register_new_resolver("round_down", math.floor)

log = logging.getLogger(__name__)


@hydra.main(
    version_base=None,
    config_path="../cfg",
)
def main(cfg: OmegaConf):
    # Resolve interpolations
    OmegaConf.resolve(cfg)

    # Set rendering backend
    sim_device = cfg.get("sim_device", None)
    if sim_device is not None:
        if isinstance(sim_device, str) and sim_device.startswith("cuda:"):
            sim_device = int(sim_device.split("cuda:")[1])
        else:
            sim_device = int(sim_device)
        os.environ["MUJOCO_GL"] = "egl"
        os.environ["MUJOCO_sim_device_ID"] = str(sim_device)
        os.environ["sim_device_ID"] = str(sim_device)
        log.info(f"Set sim_device={sim_device} from cfg.")
    else:
        # Default to egl for image tasks as we fixed EGL issues earlier
        os.environ.setdefault("MUJOCO_GL", "egl")
        log.info(f"Using MUJOCO_GL={os.environ['MUJOCO_GL']}")

    # Download assets if missing
    if "train_dataset_path" in cfg and not os.path.exists(cfg.train_dataset_path):
        url = get_dataset_download_url(cfg)
        log.info(f"Downloading dataset from {url}")
        gdown.download_folder(url=url, output=os.path.dirname(cfg.train_dataset_path))

    if "normalization_path" in cfg and not os.path.exists(cfg.normalization_path):
        url = get_normalization_download_url(cfg)
        if not os.path.exists(os.path.dirname(cfg.normalization_path)):
            os.makedirs(os.path.dirname(cfg.normalization_path))
        log.info(f"Downloading normalization from {url}")
        gdown.download(url=url, output=cfg.normalization_path, fuzzy=True)

    if (
        "base_policy_path" in cfg
        and cfg.base_policy_path
        and not os.path.exists(cfg.base_policy_path)
    ):
        url = get_checkpoint_download_url(cfg)
        if url:
            if not os.path.exists(os.path.dirname(cfg.base_policy_path)):
                os.makedirs(os.path.dirname(cfg.base_policy_path))
            log.info(f"Downloading checkpoint from {url}")
            gdown.download(url=url, output=cfg.base_policy_path, fuzzy=True)

    # Use the defined target to run the agent
    cls = hydra.utils.get_class(cfg._target_)
    agent = cls(cfg)
    agent.run()


if __name__ == "__main__":
    main()
