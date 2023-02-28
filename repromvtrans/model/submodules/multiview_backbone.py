import torch


class MultiviewBackbone(nn.Module):
    def __init__(self, hparams, in_channels=3):
        super().__init__()

        def make_disp_features():
            net = simplenet.NetFactory()
            x = net.input(in_dim=1, stride=1, activated=False)
            x = net.layer(x, 32, rate=5)
            return net.bake()

        self.disp_features = make_disp_features()

        def make_rgbd_backbone(num_channels=64, out_dim=64):
            net = simplenet.NetFactory()
            x = net.input(in_dim=64, activated=True, stride=4)
            x = net._lateral(x, out_dim=num_channels)
            x4 = x = net.block(x, "111")
            x = net.downscale(x, num_channels * 2)
            x8 = x = net.block(x, "1111")
            x = net.downscale(x, num_channels * 4)
            x = net.block(x, "12591259")
            net.tag(net.output(x, out_dim), "p4")
            x = net.upsample(x, x8, out_dim)
            net.tag(x, "p3")
            x = net.upsample(x, x4, out_dim)
            net.tag(x, "p2")
            return net.bake()

        self.rgbd_backbone = make_rgbd_backbone()
        self.reduce_channel = torch.nn.Conv2d(256, 32, 1)

    def forward(self, img_features, small_disp, robot_joint_angles=None):
        left_rgb_features = self.reduce_channel(img_features)
        disp_features = self.disp_features(small_disp)
        rgbd_features = torch.cat((disp_features, left_rgb_features), axis=1)
        outputs = self.rgbd_backbone.forward(rgbd_features)
        outputs["small_disp"] = small_disp
        return outputs
