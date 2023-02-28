from repromvtrans.dataloaders.nerf_synthetic_loader import nerf_synthetic_loader


def factory(cfg):
    if cfg.datasets.selected == "NerfSynthetic":
        return nerf_synthetic_loader(cfg)
    else:
        raise NotImplementedError(f"Dataset {cfg.dataset.name}")
