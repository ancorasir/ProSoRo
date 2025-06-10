#!/usr/bin/env python

import argparse
import os
import shutil
import torch
import yaml
import pytorch_lightning as L
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from models.data_module import DataModule
from models.mvae import MVAE


def train(
    lr: float = 1e-5,
    batch_size: int = 128,
    max_epochs: int = 1000,
) -> None:
    """Train the MVAE model.

    The trained model is saved in the './models/pths' folder.
    The log files are saved in the './lightning_logs' folder.

    Args:
        lr: the learning rate
        batch_size: the batch size
        max_epochs: the maximum number of epochs
    """

    with open("./config/model.yaml", "r") as f:
        config = yaml.load(f.read(), Loader=yaml.Loader)
    object = config["object"]
    x_dim = config["x_dim"]
    z_dim = config["z_dim"]
    h1_dim = config["h1_dim"]
    h2_dim = config["h2_dim"]
    recon_pred_scale = config["recon_pred_scale"]
    kl_coef = config["kl_coef"]
    z_coef = config["z_coef"]

    ## DataModule
    dm = DataModule(object=object, batch_size=batch_size, num_workers=8)
    dm.setup()

    ## Callbacks
    trainer = L.Trainer(
        max_epochs=max_epochs,
        accelerator="auto",
        devices=1,
        strategy="auto",
        callbacks=[
            ModelCheckpoint(save_weights_only=True),
            LearningRateMonitor("epoch"),
        ],
    )

    ## Instantiate
    model = MVAE(
        x_dim=x_dim,
        h1_dim=h1_dim,
        h2_dim=h2_dim,
        z_dim=z_dim,
        lr=lr,
        recon_pred_scale=recon_pred_scale,
        z_coef=z_coef,
        kl_coef=kl_coef,
    )

    ## Training
    trainer.fit(model, dm)

    ## Evaluation
    model.eval()

    ## Save the model
    if not os.path.exists("models/pths"):
        os.makedirs("models/pths")

    torch.save(
        model.state_dict(),
        "models/pths/mvae_"
        + object
        + "_"
        + str(recon_pred_scale)
        + "_"
        + str(z_coef)
        + "_"
        + str(kl_coef)
        + "_"
        + str(z_dim)
        + ".pth",
    )

    ## Rename log folder
    version_folder = "lightning_logs/version_" + str(trainer.logger.version)
    new_version_folder = (
        "./lightning_logs/mvae_"
        + object
        + "_"
        + str(recon_pred_scale)
        + "_"
        + str(z_coef)
        + "_"
        + str(kl_coef)
        + "_"
        + str(z_dim)
    )
    if os.path.exists(new_version_folder):
        shutil.rmtree(new_version_folder)
    os.rename(version_folder, new_version_folder)


if __name__ == "__main__":
    # Parse the arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--lr", type=float, default=1e-5, help="Learning rate (default: 1e-5)."
    )
    parser.add_argument(
        "--batch-size", type=int, default=128, help="Batch size (default: 128)."
    )
    parser.add_argument(
        "--max-epochs", type=int, default=1000, help="Max epochs (default: 1000)."
    )
    args = parser.parse_args()

    train(
        lr=args.lr,
        batch_size=args.batch_size,
        max_epochs=args.max_epochs,
    )
