from omegaconf import OmegaConf
import torch

import pytorch_lightning as pl
import os

from pytorch_lightning.loggers import WandbLogger

from repromvtrans.dataloaders import data_factory
from repromvtrans.runner import Runner


def main():
    cfg = OmegaConf.load("config/config.yaml")
    cmd_cfg = OmegaConf.from_cli()
    cfg = OmegaConf.merge(cfg, cmd_cfg)

    print(OmegaConf.to_yaml(cfg))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    pl.seed_everything(cfg.hyperparameters.seed, workers=True)
    os.environ["WAND_CACHE_DIR"] = os.path.join(cfg.wandb.dir, "cache")

    wandb_logger = WandbLogger(
        save_dir=cfg.wandb.dir,
        project=cfg.wandb.project,
        name=cfg.wandb.experiment_name,
        log_model="all" if cfg.wandb.log else None,
        offline=not cfg.wandb.log,
        # Keyword args passed to wandb.init()
        entity=cfg.wandb.entity,
        config=OmegaConf.to_object(cfg),
    )

    [train_loader, val_loader, test_loader] = data_factory.factory(cfg)
    print(f"batch counts -> {len(train_loader)}:{len(val_loader)}:{len(test_loader)}")

    runner = Runner(cfg)
    trainer = pl.Trainer(
        max_epochs=cfg.train.epochs,
        default_root_dir=cfg.wandb.save_dir,
        logger=wandb_logger,
        accelerator="cpu",
        strategy="ddp_find_unused_parameters_false",
        gpus=torch.cuda.device_count(),
    )

    trainer.fit(runner, train_loader, val_loader)

    trainer.test(runner, test_loader)


if __name__ == "__main__":
    main()
