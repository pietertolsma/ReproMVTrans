import torch.nn as nn
import torch


class UpsamplingBlock(nn.Module):
    def __init__(self, batch_size=32, out_channels=64, use_bias=True, interpolate=True):
        super().__init__()
        self.interpolate = interpolate
        self.convolute = nn.Sequential(
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=use_bias,
            ),
            nn.GroupNorm(batch_size, out_channels),
            nn.ReLU(),
        )

    def forward(self, features):
        _, _, H, W = features.shape
        x = self.convolute(features)
        if not self.interpolate:
            return x
        return nn.functional.interpolate(
            x, [H * 2, W * 2], mode="bilinear", align_corners=True
        )


class PanopticSegmentationFPN(nn.Module):
    def __init__(self, batch_size, out_channels, classes=2):
        super().__init__()
        self.classes = classes
        self.block = UpsamplingBlock(batch_size=batch_size, out_channels=out_channels)

        self.upsample_1_4 = UpsamplingBlock(
            batch_size=batch_size, out_channels=out_channels, interpolate=False
        )
        self.upsample_1_8 = nn.Sequential(self.block)
        self.upsample_1_16 = nn.Sequential(self.block, self.block)

        self.final_convolute = nn.Conv2d(
            in_channels=out_channels, out_channels=classes, kernel_size=1, bias=True
        )

        self.softmax = nn.Softmax(dim=1)

    def forward(self, in_features):
        # features tagged with p4, p3 and p2
        # being sizes 1/16th, 1/8th and 1/4th
        # of original size respectively.
        _, _, H, W = in_features["p2"].shape
        p2_up = self.upsample_1_4(in_features["p2"])
        p3_up = self.upsample_1_8(in_features["p3"])
        p4_up = self.upsample_1_16(in_features["p4"])

        summed = p4_up + p3_up + p2_up
        x = self.final_convolute(summed)
        x = nn.functional.interpolate(
            x, [H * 4, W * 4], mode="bilinear", align_corners=True
        )
        # Dim should be B x NClasses x 800 x 800
        return self.softmax(x)
