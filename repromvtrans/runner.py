import pytorch_lightning as pl
import torch

from repromvtrans.model import model_factory

from torchmetrics import PeakSignalNoiseRatio


class Runner(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.save_hyperparameters()
        self.cfg = cfg
        self.model = model_factory.factory(cfg)

        self.cam_intr = torch.Tensor(
            [
                [
                    [cfg.datasets.camera.fx, 0, cfg.datasets.camera.px],
                    [0, cfg.datasets.camera.fy, cfg.datasets.camera.py],
                    [0, 0, 1],
                ]
            ]
        )

        self.loss_fn = torch.nn.HuberLoss()

        self.train_accuracy = PeakSignalNoiseRatio()
        self.val_accuracy = PeakSignalNoiseRatio()
        self.test_accuracy = PeakSignalNoiseRatio()

        self.device == "cuda" if torch.cuda.is_available() else "cpu"

    def forward(self, data):
        return self.model(data)

    def configure_optimizers(self):
        if self.cfg.optimize.optimizer == "Adam":
            optimizer = torch.optim.Adam(
                self.model.parameters(), lr=self.cfg.optimize.lr
            )
        else:
            raise NotImplementedError(f"Optimizer {self.cfg.optimizer}")
        return optimizer

    def _step(self, batch):
        (imgs, cams), y = batch
        depth = self.model(imgs, cams, self.cam_intr, device=self.device)
        loss = depth.compute_loss(y, device=self.device)
        # loss = self.loss_fn(y_hat, y)
        return loss, depth.depth_pred

    def training_step(self, batch, batch_idx):
        loss, y_hat = self._step(batch)
        self.train_accuracy(y_hat, batch[1])

        self.log("train/loss_step", loss)
        self.log("train/acc_step", self.train_accuracy)

        return loss

    def validation_step(self, batch, batch_idx):
        loss, y_hat = self._step(batch)
        self.val_accuracy(y_hat, batch[1])

        self.log("val/loss_step", loss)
        self.log("val/acc_step", self.val_accuracy)

        return loss

    def test_step(self, batch, batch_idx):
        loss, y_hat = self._step(batch)
        self.test_accuracy(y_hat, batch[1])

        self.log("test/loss_step", loss)
        self.log("test/acc_step", self.test_accuracy)

        return loss

    def on_train_epoch_end(self) -> None:
        self.log("train/acc", self.train_accuracy.compute())
        self.train_accuracy.reset()

    def on_validation_epoch_end(self) -> None:
        self.log("val/acc", self.val_accuracy.compute())
        self.val_accuracy.reset()
