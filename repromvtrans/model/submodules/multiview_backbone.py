import torch
import torch.nn as nn

from repromvtrans.utils.simplenet import NetFactory


class MultiviewBackbone(nn.Module):
    def __init__(self):
        super().__init__()

        self.disp_network = nn.Sequential(
            nn.Conv2d(1, 32, dilation=5, stride=1, kernel_size=3, padding=5),
            nn.Conv2d(32, 32, dilation=5, stride=1, kernel_size=3, padding=5),
        )

        net = NetFactory()
        x = net.input(in_dim=64, activated=True, stride=4)
        x = net._lateral(x, out_dim=64)
        x4 = x = net.block(x, "111")
        x = net.downscale(x, 64 * 2)
        x8 = x = net.block(x, "1111")
        x = net.downscale(x, 64 * 4)
        x = net.block(x, "12591259")
        net.tag(net.output(x, 64), "p4")
        x = net.upsample(x, x8, 64)
        net.tag(x, "p3")
        x = net.upsample(x, x4, 64)
        net.tag(x, "p2")

        self.rgbd_network = net.bake()

    def forward_disparity(self, disp):
        return self.disp_network(disp)

    def forward_rgbd(self, rgbd):
        return self.rgbd_network(rgbd)

    def forward(self, img_features, small_disp):
        disp_features = self.forward_disparity(small_disp)
        rgbd_features = torch.cat([disp_features, img_features])
        output = self.forward_rgbd(rgbd_features, axis=1)
        return output
