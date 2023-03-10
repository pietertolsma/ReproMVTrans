# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

import numpy as np

import torch.nn as nn
import torchvision.models as models


class ResnetEncoder(nn.Module):
    """Pytorch module for a resnet encoder"""

    def __init__(self, num_layers, pretrained, num_input_images=1):
        super(ResnetEncoder, self).__init__()

        self.num_ch_enc = np.array([64, 64, 128, 256, 512])

        resnets = {
            18: models.resnet18,
            34: models.resnet34,
            50: models.resnet50,
        }

        weights = {
            18: models.ResNet18_Weights,
            34: models.ResNet34_Weights,
            50: models.ResNet50_Weights,
        }

        if num_layers not in resnets:
            raise ValueError(
                "{} is not a valid number of resnet layers".format(num_layers)
            )

        self.encoder = resnets[num_layers](
            weights=weights[num_layers] if pretrained is True else None
        )

        if num_layers > 34:
            self.num_ch_enc[1:] *= 4

    def forward(self, x):
        self.features = []

        x = self.encoder.conv1(x)
        x = self.encoder.bn1(x)
        self.features.append(self.encoder.relu(x))
        self.features.append(
            self.encoder.layer1(self.encoder.maxpool(self.features[-1]))
        )
        self.features.append(self.encoder.layer2(self.features[-1]))
        self.features.append(self.encoder.layer3(self.features[-1]))
        self.features.append(self.encoder.layer4(self.features[-1]))

        return self.features  # the feature maps is activated by relu
