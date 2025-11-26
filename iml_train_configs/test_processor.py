import torch
import numpy as np
from dataclasses import dataclass
from typing import Dict

# load dataset
from lerobot.envs.utils import preprocess_observation
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.policies.fact.configuration_fact import FACTConfig

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.utils.robot_utils import busy_wait

from lerobot.processor.quaternion_observations_processor import RotTransProcessorStep
from subset_dataset import SubsetStateActionDataset

import torch
from torch.utils.data import DataLoader
from lerobot.utils.constants import OBS_STATE, ACTION
from lerobot.processor.core import TransitionKey

# Translate all of Chinese to English

STATE_KEEP_NAMES = [
    "ee_x_l", "ee_y_l", "ee_z_l",
    "ee_qx_l", "ee_qy_l", "ee_qz_l", "ee_qw_l",
    "force_x_l", "force_y_l", "force_z_l",
    "torque_x_l", "torque_y_l", "torque_z_l",
    "ee_x_r", "ee_y_r", "ee_z_r",
    "ee_qx_r", "ee_qy_r", "ee_qz_r", "ee_qw_r",
    "force_x_r", "force_y_r", "force_z_r",
    "torque_x_r", "torque_y_r", "torque_z_r",
    "stage",
]

ACTION_KEEP_NAMES = [
    "ee_x_l", "ee_y_l", "ee_z_l",
    "ee_qx_l", "ee_qy_l", "ee_qz_l", "ee_qw_l",
    "gripper_l",
    "ee_x_r", "ee_y_r", "ee_z_r",
    "ee_qx_r", "ee_qy_r", "ee_qz_r", "ee_qw_r",
    "gripper_r",
]

def main():
    # 1. Load Dataset
    repo_id = "leledeyuan/hanging-tshirt"
    print(f"Loading dataset: {repo_id} ...")
    
    try:
        dataset = LeRobotDataset(repo_id=repo_id)
    except Exception as e:
        print(f"❌ Dataset loading failed: {e}")
        print("Hint: Please check your network or dataset_repo_id.")
        return

    dataset = SubsetStateActionDataset(dataset, STATE_KEEP_NAMES, ACTION_KEEP_NAMES)

    # 2. Create DataLoader to get a Batch
    batch_size = 2
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
    )
    
    # First batch
    data_iter = iter(data_loader)
    raw_batch = next(data_iter)
    
    print("\n🎯 Dataset loaded successfully.")
    
    # 3. Format Conversion (Bridge)
    config = FACTConfig()
    config.phase_num = 3
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=config,
    )
    
    # 4. Initialize your Processor
    
    print("\n🔄 Running Processor...")
    try:
        # 5. Process the Batch
        output_transition = preprocessor(raw_batch)
        
        print("\n🎉 Processor succeeded.")
        
        print("Processed Keys:", list(output_transition.keys()))

        # Print original and processed observation state shapes
        print("observation rotation:", output_transition["observation_rotation"])
        print("phase", output_transition["phase"])


    except Exception as e:
        print(f"\n❌ Processor error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()