#!/usr/bin/env python

# Copyright 2024 Tony Z. Zhao and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Action Chunking Transformer Policy

As per Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware (https://huggingface.co/papers/2304.13705).
The majority of changes here involve removing unused code, unifying naming, and adding helpful comments.
"""

from collections.abc import Callable
import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn

from lerobot.policies.factphase.configuration_factphase import FACTPhaseConfig
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.utils.constants import ACTION, OBS_ENV_STATE, OBS_IMAGES, OBS_STATE
from lerobot.policies.fact.modeling_fact import FACTPolicy


class FACTPhasePolicy(PreTrainedPolicy):
    """
    A Phase classification policy based on the Force-based Action Chunking Transformer (FACT) architecture.
    """

    config_class = FACTPhaseConfig
    name = "factphase"

    def __init__(
        self,
        config: FACTPhaseConfig,
    ):
        """
        Args:
            config: Policy configuration class instance or None, in which case the default instantiation of
                    the configuration class is used.
        """
        super().__init__(config)
        config.validate_features()
        self.config = config

        self.model = FACTPhase(config)
        
        # load each class numbers for calculating loss weights
        class_nums = config.class_nums
        if len(class_nums) != config.phase_num:
            raise ValueError("Length of class_nums must be equal to phase_num.")

        class_weights = 1.0 / torch.tensor(class_nums, dtype=torch.float)
        class_weights = class_weights / class_weights.sum()
        self.class_weights = class_weights.to(config.device)
        self.loss = nn.CrossEntropyLoss(weight=self.class_weights)

        self.reset()

    def get_optim_params(self) -> dict:
        # TODO(aliberts, rcadene): As of now, lr_backbone == lr
        # Should we remove this and just `return self.parameters()`?
        return [
            {
                "params": [
                    p
                    for n, p in self.named_parameters()
                    if not n.startswith("model.backbone") and p.requires_grad
                ]
            },
            {
                "params": [
                    p
                    for n, p in self.named_parameters()
                    if n.startswith("model.backbone") and p.requires_grad
                ],
                "lr": self.config.optimizer_lr_backbone,
            },
        ]

    @torch.no_grad()
    def predict_phase(self, batch: dict[str, Tensor]) -> Tensor:
        """Predict phases given a batch of observations."""
        self.eval()

        if self.config.image_features:
            batch = dict(batch)  # shallow copy so that adding a key doesn't modify the original
            batch[OBS_IMAGES] = [batch[key] for key in self.config.image_features]

        phase_pred = self.model(batch)[0]
        # Softmax to get probabilities
        phase_probs = torch.softmax(phase_pred, dim=-1)
        # top-k phase predictions
        values, indices = torch.topk(phase_probs, k=self.config.pred_topk, dim=-1)

        return values, indices
    
    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor]) -> Tensor:
        """
        This method is required by PreTrainedPolicy but not used for Phase classification.
        The reward classifier is not an actor and does not select actions.
        """
        raise NotImplementedError("Reward classifiers do not select actions")
    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor]) -> Tensor:
        """
        This method is required by PreTrainedPolicy but not used for Phase classification.
        The reward classifier is not an actor and does not produce action chunks.
        """
        raise NotImplementedError("Reward classifiers do not predict action chunks")

    def reset(self):
        """
        This method is required by PreTrainedPolicy but not used for Phase classification.
        The reward classifier is not an actor and does not select actions.
        """
        pass

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict]:
        """Run the batch through the model and compute the loss for training or validation."""
        if self.config.image_features:
            batch = dict(batch)  # shallow copy so that adding a key doesn't modify the original
            batch[OBS_IMAGES] = [batch[key] for key in self.config.image_features]

        phase_pred = self.model(batch)


        if self.config.phase_num is not None:
            batch_phase = batch["phase"]  # (B, 1)
            batch_phase = batch_phase.flatten().long()

        # Compute loss.
        loss = self.loss(phase_pred, batch_phase)
        loss_dict = {"loss": loss.item()}
        
        return loss, loss_dict


class FACTPhase(nn.Module):

    def __init__(self, config: FACTPhaseConfig):
        # BERT style VAE encoder with input tokens [cls, robot_state, *action_sequence].
        # The cls token forms parameters of the latent's distribution (like this [*means, *log_variances]).
        super().__init__()
        self.config = config
        fact_backbone_path = config.pretrained_fact_path
        fact_policy = None
        if fact_backbone_path is not None:
            print(f"Loading FACTPhase backbone from {fact_backbone_path}")
            fact_policy = FACTPolicy.from_pretrained(fact_backbone_path)
        else:
            raise ValueError("Please provide a pretrained FACT model path for FACTPhasePolicy.")
        fact_model = fact_policy.model
        
        # image backbone and conv1
        self.backbone = fact_model.backbone
        self.conv1 = fact_model.encoder_img_feat_input_proj
        self.glb_ave_pool = nn.AdaptiveAvgPool2d((1,1))

        self.encoder_robot_wrench_input_proj = fact_model.encoder_robot_wrench_input_proj
        self.encoder_robot_state_input_proj = fact_model.encoder_robot_state_input_proj
        # frozen params from pretrained FACT
        for param in self.backbone.parameters():
            param.requires_grad = False
        for param in self.conv1.parameters():
            param.requires_grad = False
        for param in self.encoder_robot_wrench_input_proj.parameters():
            param.requires_grad = False
        for param in self.encoder_robot_state_input_proj.parameters():
            param.requires_grad = False

        # MLP projection from concatenated features to Num phases
        hidden_dim = config.dim_model
        self.mlp_proj = nn.Sequential(
            nn.Linear( hidden_dim* 6, hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(hidden_dim, config.phase_num)
        )


    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, tuple[Tensor, Tensor] | tuple[None, None]]:
        """A forward pass through the FACTPhase model.

        Returns:
            Phase logits with shape (B, phase_num).
        """

        batch_size = batch[OBS_IMAGES][0].shape[0] if OBS_IMAGES in batch else batch[OBS_ENV_STATE].shape[0]

        # prepare the force and rotation as part of the action input, using the 6D representation for rotations instead of original quaternion
        # select states
        position_indices = [0,1,2,                  # left pos
                        13,14,15,]              # right pos
        wrench_indices = [7,8,9,10,11,12,       # left wrench
                          20,21,22,23,24,25]    # right wrench

        batch_position = batch[OBS_STATE][:, position_indices]  # (B, 6)
        batch_wrench = batch[OBS_STATE][:, wrench_indices]  # (B, 12)
        batch_rot_state = batch["observation_rotation"] # (B, 12)
        # Add noise to the position, wrench for data augmentation during training
        if self.training:
            # poisition noise
            noise = torch.randn_like(batch_position) * 0.001  # Gaussian noise with stddev of 0.001
            batch_position = batch_position + noise
            # wrench noise
            noise = torch.randn_like(batch_wrench) * 0.1  # Gaussian noise
            scale_noise = torch.rand(1).to(batch_wrench.device) * 0.2  + 0.9 # random scaling between 0.9 and 1.1
            bias_noise = torch.randn_like(batch_wrench) * 0.02  # Gaussian bias noise
            batch_wrench = batch_wrench * scale_noise + noise + bias_noise
            
        batch_pose = torch.cat([batch_position, batch_rot_state], dim=-1)  # (B, 18)
        # Prepare mlp inputs.
        mlp_in_tokens = []
        # Robot state token.
        if self.config.robot_state_feature:
            mlp_in_tokens.append(self.encoder_robot_state_input_proj(batch_pose))
        if self.config.wrench_dim is not None:
            mlp_in_tokens.append(self.encoder_robot_wrench_input_proj(batch_wrench))

        if self.config.image_features:
            # For a list of images, the H and W may vary but H*W is constant.
            # NOTE: If modifying this section, verify on MPS devices that
            # gradients remain stable (no explosions or NaNs).
            for img in batch[OBS_IMAGES]:
                cam_features = self.backbone(img)["feature_map"]
                cam_features = self.conv1(cam_features)
                # Global Average Pooling
                gap_output = self.glb_ave_pool(cam_features)
                gap_output = gap_output.view(batch_size, -1)
                mlp_in_tokens.append(gap_output)

        # Concatenate all input tokens for the MLP.
        encoder_in_tokens = torch.cat(mlp_in_tokens, dim=-1)  # (B, C_total)

        # MLP projection to phase logits
        phase_logits = self.mlp_proj(encoder_in_tokens)  # (B, phase_num)
       
        return phase_logits



def get_activation_fn(activation: str) -> Callable:
    """Return an activation function given a string."""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(f"activation should be relu/gelu/glu, not {activation}.")
