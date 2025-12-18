import torch
import numpy as np
from dataclasses import dataclass
from typing import Dict
from torch import nn

from lerobot.policies.fact.configuration_fact import FACTConfig
from lerobot.policies.fact.modeling_fact import FACTPolicy


# Translate all of Chinese to English

def main():
    pretrained_fact_path = "/home/dayuan/nas/models/fact_flatten_mixed_pwms_50chunk_2dec/checkpoints/480000/pretrained_model"
    policy = FACTPolicy.from_pretrained(
        pretrained_fact_path,
    )
    model = policy.model
    backbone = model.backbone.to('cpu')
    conv11 = model.encoder_img_feat_input_proj.to('cpu')
    # print(backbone)

    num_phases = 11

    # Test forward pass with dummy data
    batch_size = 8
    image_height = 480
    image_width = 640
    hidden_dim = 512
    dummy_images = torch.randn(batch_size, 3, image_height, image_width).to('cpu')
    image_outputs = conv11(backbone(dummy_images)['feature_map'])
    print("Backbone output shape:", image_outputs.shape)

    # Global Average Pooling
    global_ave_pool = nn.AdaptiveAvgPool2d((1,1))
    gap_output = global_ave_pool(image_outputs)
    gap_output = gap_output.view(batch_size, -1)
    print("GAP output shape:", gap_output.shape)

    # Dummy wrench data
    dummy_wrench = torch.randn(batch_size, 12).to('cpu')
    print("Dummy wrench shape:", dummy_wrench.shape)
    wrench_proj = model.encoder_robot_wrench_input_proj.to('cpu')
    wrench_output = wrench_proj(dummy_wrench)
    print("Wrench projected shape:", wrench_output.shape)

    # concatenate 4 images and wrench
    images_wrench = torch.cat([wrench_output, gap_output, gap_output, gap_output, gap_output], dim=-1)

    # MLP projection from concatenated features to Num phases
    mlp_proj = nn.Sequential(
        nn.Linear( hidden_dim* 5, hidden_dim),
        nn.ReLU(),
        nn.Dropout(0.1),
        nn.Linear(hidden_dim, num_phases)
    ).to('cpu')

    phase_logits = mlp_proj(images_wrench)
    print("Phase logits shape:", phase_logits.shape)
    phase_probs = torch.softmax(phase_logits, dim=-1)
    print("Phase probabilities shape:", phase_probs.shape)
    predicted_phases = torch.argmax(phase_probs, dim=-1)
    print("Predicted phases shape:", predicted_phases.shape)
    print("Predicted phases:", predicted_phases)

    # print model layers
    # for name, module in backbone.named_modules():
    #     print(name, module)
    


if __name__ == "__main__":
    main()