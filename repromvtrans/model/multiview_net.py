import torch
import torch.nn as nn

from repromvtrans.model.submodules.depth_decoder import DepthHybridDecoder
from repromvtrans.model.submodules.psm_submodule import PSMFeatureExtraction
from repromvtrans.model.submodules.resnet_encoder import ResnetEncoder
from repromvtrans.utils.epipolar_ops import homo_warping
from repromvtrans.utils.transform_ops import scale_basis


class MVNet(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.stage1_scale = cfg.model.mvnet.parameters.stage1_scale
        self.stage2_scale = cfg.model.mvnet.parameters.stage2_scale
        self.stage3_scale = cfg.model.mvnet.parameters.stage3_scale

        self.ndepths = cfg.model.mvnet.parameters.ndepths
        self.depth_min = cfg.model.mvnet.parameters.depth_min
        self.depth_max = cfg.model.mvnet.parameters.depth_max
        self.depth_interval = (self.depth_max - self.depth_min) / (self.ndepths - 1)

        self.depth_cands = (
            torch.arange(0, self.ndepths, requires_grad=False)
            .reshape(1, -1)
            .to(torch.float32)
            * self.depth_interval
            + self.depth_min
        )

        self.matchingFeature = PSMFeatureExtraction()
        self.semanticFeature = ResnetEncoder(
            cfg.model.mvnet.parameters.resnet, pretrained=True
        )

        self.costRegNet = DepthHybridDecoder(
            self.semanticFeature.num_ch_enc,
            num_output_channels=1,
            use_skips=True,
            ndepths=self.ndepths,
            depth_max=self.depth_max,
            IF_EST_transformer=cfg.model.mvnet.parameters.transformer,
        )

    def forward(
        self,
        imgs,
        cam_poses,
        pre_costs=None,
        pre_cam_poses=None,
        mode="train",
        device="cpu",
    ):
        assert (
            len(cam_poses.shape) == 4
        ), f"Expected cam_poses to be 4-dimensional, but is {len(cam_poses)}"

        imgs = 2 * (imgs / 255.0) - 1
        assert len(imgs.shape) == 5, "Imgs must be BxVxCxHxW"

        batch_size, views_num, _, height_img, width_img = imgs.shape
        assert views_num > 1, f"View count should be larger than 1, but is {views_num}"

        height = height_img // 4
        width = width_img // 4

        matching_features = self.matchingFeature(
            imgs.view(batch_size * views_num, 3, height_img, width_img)
        )  # 32 channels
        matching_features = matching_features.view(
            batch_size, views_num, -1, height, width
        )
        matching_features = matching_features.permute(
            1, 0, 2, 3, 4
        ).contiguous()  # VxBxCxHxW

        semantic_features = self.semanticFeature(
            imgs[:, 0].view(batch_size, -1, height_img, width_img)
        )  # select reference views

        cam_intr_stage1 = scale_basis(cam_poses[0], scale=1.0 / self.stage1_scale)
        depth_values = (
            self.depth_cands.view(1, self.ndepths, 1, 1)
            .repeat(batch_size, 1, 1, 1)
            .to(imgs.dtype)
            .to(imgs.device)
        )

        # Project the features on the cost volume based on the epipolar lines.
        cost_volume = self.get_costvolume(
            matching_features, cam_poses, cam_intr_stage1, depth_values, device
        )

        outputs, cur_costs, cur_cam_poses = self.costRegNet(
            costvolumes=[cost_volume],
            semantic_features=semantic_features,
            cam_poses=[],
            cam_intr=cam_intr_stage1,
            depth_values=depth_values,
            depth_min=self.depth_min,
            depth_interval=self.depth_interval,
            pre_costs=pre_costs,
            pre_cam_poses=pre_cam_poses,
            mode=mode,
        )

        # # obtain the small disparity output for the reference image.
        # small_disp_output = outputs[("depth", 0, 2)]  # B x 1 x H x W
        # assert (
        #     len(small_disp_output.shape) == 4
        # ), f"Expecting depth to be Nx1xHxW, but got {small_disp_output.shape}"

        return outputs[("depth", 0, 0)]

    def get_costvolume(self, features, cam_poses, cam_intr, depth_values, device="cpu"):
        """
        return cost volume, [ref_feature, warped_feature] concat
        :param features: middle one is ref feature, others are source features
        :param cam_poses:
        :param cam_intr:
        :param depth_values:
        :return:
        """
        num_views = len(features)
        ref_feature = features[0]
        ref_cam_pose = cam_poses[:, 0, :, :]
        ref_extrinsic = torch.inverse(ref_cam_pose)

        # step 2. differentiable homograph, build cost volume
        ref_volume = ref_feature.unsqueeze(2).repeat(1, 1, self.ndepths, 1, 1)
        costvolume = (
            torch.zeros_like(ref_volume).to(ref_volume.dtype).to(ref_volume.device)
        )
        for view_i in range(num_views):
            if view_i == 0:
                continue
            src_fea = features[view_i]
            src_cam_pose = cam_poses[:, view_i, :, :]
            src_extrinsic = torch.inverse(src_cam_pose)

            # warpped features
            src_proj_new = src_extrinsic.clone()
            ref_proj_new = ref_extrinsic.clone()

            ref_proj_new = ref_proj_new.to("cpu")
            cam_intr = cam_intr.to("cpu")
            ref_extrinsic = ref_extrinsic.to("cpu")
            src_extrinsic = src_extrinsic.to("cpu")

            src_proj_new[:, :3, :4] = torch.matmul(cam_intr, src_extrinsic[:, :3, :4])
            ref_proj_new[:, :3, :4] = (cam_intr @ ref_extrinsic[:, :3, :4]).clone()
            ref_proj_new = ref_proj_new.to(device)
            cam_intr = cam_intr.to(device)
            ref_extrinsic = ref_extrinsic.to(device)
            src_extrinsic = src_extrinsic.to(device)

            warped_volume = homo_warping(
                src_fea, src_proj_new, ref_proj_new, depth_values
            )
            x = torch.cat([ref_volume, warped_volume], dim=1)
            x = self.pre0(x)
            x = x + self.pre2(self.pre1(x))
            costvolume = costvolume + x

        # aggregate multiple feature volumes by variance
        costvolume = costvolume / (num_views - 1)
        del warped_volume
        del x
        return costvolume
