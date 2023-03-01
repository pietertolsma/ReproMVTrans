import os
import json
import torch
from torch import Tensor
from torch.utils.data import random_split
from torch.utils.data import Dataset, DataLoader
import numpy as np

from PIL import Image


def select_n_nearest(target_index, cameras: list[np.ndarray], n) -> list[int]:
    res = []
    normal = cameras[target_index][:3, 2]
    for _, cam in enumerate(cameras):
        source_normal = cam[:3, 2]
        dot = np.dot(normal, source_normal).clip(0, 1)
        angle = np.arccos(dot)
        res.append(angle)

    return np.argsort(res)[:n]


class NerfSyntheticParser:
    def __init__(self, cfg):
        assert os.path.exists(
            cfg.datasets.nerf_synthetic.root
        ), "Could not find the data folder."

        self.data_path = cfg.datasets.nerf_synthetic.root
        self.cfg = cfg

        self.config = json.load(open(self.data_path + "/transforms.json", "r"))
        self.frame_count = len(self.config["frames"])

    def get_image(self, frame_index):
        config = self.config
        assert frame_index < self.frame_count, "Index out of bounds"

        frame = config["frames"][frame_index]
        image = Image.open(f"{self.data_path}{frame['file_path'][1:]}.png")
        return Tensor(np.array(image)).permute(2, 0, 1)[:3, :, :]

    def get_depth(self, frame_index):
        config = self.config
        assert frame_index < self.frame_count, "Index out of bounds"

        frame = config["frames"][frame_index]
        image = (
            Image.open(f"{self.data_path}{frame['file_path'][1:]}_depth_0001.png")
            .convert("L")
            .resize((200, 200))
        )
        return Tensor(np.array(image)) / 255

    def get_cam_transform(self, frame_index: int):
        config = self.config
        assert frame_index < self.frame_count, "Index out of bounds"

        frame = config["frames"][frame_index]
        tf_mat = Tensor(frame["transform_matrix"])

        return tf_mat

    def get_all_transforms(self):
        return [
            np.array(self.config["frames"][i]["transform_matrix"])
            for i in range(self.frame_count)
        ]


class NerfSyntheticDataset(Dataset):
    def __init__(self, cfg):
        self.parser = NerfSyntheticParser(cfg)
        self.cameras = self.parser.get_all_transforms()

        self.view_count = cfg.model.mvnet.parameters.view_count

    def __len__(self):
        return len(self.cameras)

    def __getitem__(self, idx):
        indices = select_n_nearest(idx, self.cameras, self.view_count)

        imgs = torch.stack([self.parser.get_image(i) for i in indices])
        imgs = imgs.view(self.view_count, 3, imgs.shape[2], imgs.shape[3])
        cams = torch.stack([self.parser.get_cam_transform(i) for i in indices])
        cams = cams.view(self.view_count, 4, 4)
        depth = self.parser.get_depth(idx)

        return (imgs, cams), depth


def nerf_synthetic_loader(cfg):
    shares = [
        cfg.datasets.nerf_synthetic.train_share,
        cfg.datasets.nerf_synthetic.val_share,
        cfg.datasets.nerf_synthetic.test_share,
    ]

    ds = NerfSyntheticDataset(cfg)
    split = random_split(
        dataset=ds,
        lengths=shares,
        generator=torch.Generator().manual_seed(cfg.hyperparameters.seed),
    )

    print(f"{len(split[0])}:{len(split[1])}:{len(split[2])}")

    return [
        DataLoader(
            split[i],
            batch_size=cfg.train.batch_size,
            num_workers=cfg.train.num_workers,
            shuffle=False,
        )
        for i in range(3)
    ]
