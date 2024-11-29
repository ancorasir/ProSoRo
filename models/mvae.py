#!/usr/bin/env python

from typing import Any, List
import torch
from torch import nn, Tensor
from pytorch_lightning import LightningModule
from torch.nn import functional as F


class MVAE(LightningModule):
    """Multi-modal Variational Autoencoder (MVAE)

    MVAE is a multi-modal learning model that can learn the shared information among different modalities.
    In this work, the MVAE model is to represent the motion, force, and shape of ProSoRo.

    Input: motion, force, and shape
    Output: motion, force, and shape

    motion: the motion (Dx, Dy, Dz, Rx, Ry, Rz) of the top surface
    force: the forces and torques (Fx, Fy, Fz, Tx, Ty, Tz) at the bottom surface
    shape: the displacements (Dx, Dy, Dz) of the nodes on the surface
    """

    def __init__(
        self,
        x_dim: list = [6, 6, 2136],
        h1_dim: list = [8, 8, 512],
        h2_dim: list = [16, 16, 128],
        z_dim: int = 32,
        lr: float = 1e-4,
        recon_pred_scale: float = 1,
        z_coef: float = 1,
        kl_coef: float = 1,
        **kwargs,
    ) -> None:
        """Initialize the MVAE model.

        Args:
            x_dim: the dimension of the input data
            h1_dim: the dimension of the hidden layer 1
            h2_dim: the dimentsion of the hidden layer 2
            z_dim: dim of z space
            lr: learning rate for Adam
            recon_pred_scale: the scale for the reconstruction and prediction loss
            z_coef: the coefficient for the latent loss
            kl_coef: the coefficient for the KL divergence
        """

        # Call the super constructor
        super().__init__()

        # Save hyperparameters
        self.save_hyperparameters()
        self.lr = lr
        self.z_dim = z_dim
        self.x_dim = x_dim
        self.h1_dim = h1_dim
        self.h2_dim = h2_dim
        self.recon_pred_scale = recon_pred_scale
        self.z_coef = z_coef
        self.kl_coef = kl_coef

        # Define the model architecture
        for i in range(len(self.x_dim)):
            # Encoder
            setattr(
                self,
                f"encoder_{i}",
                nn.Sequential(
                    nn.Linear(self.x_dim[i], self.h1_dim[i]),
                    nn.ReLU(),
                    nn.Linear(self.h1_dim[i], self.h2_dim[i]),
                ),
            )
            # Decoder
            setattr(
                self,
                f"decoder_{i}",
                nn.Sequential(
                    nn.Linear(self.z_dim, self.h2_dim[i]),
                    nn.ReLU(),
                    nn.Linear(self.h2_dim[i], self.h1_dim[i]),
                    nn.ReLU(),
                    nn.Linear(self.h1_dim[i], self.x_dim[i]),
                ),
            )
            # mu and var
            setattr(self, f"fc_mu_{i}", nn.Linear(self.h2_dim[i], self.z_dim))
            setattr(self, f"fc_var_{i}", nn.Linear(self.h2_dim[i], self.z_dim))

    @staticmethod
    def pretrained_weights_available():
        """Check if the pretrained weights are available.

        Returns:
            bool: True if the pretrained weights are available, False otherwise.
        """

        pass

    def from_pretrained(self, checkpoint_name):
        """Load the pretrained weights.

        Args:
            checkpoint_name: the name of the checkpoint file.

        Returns:
            None
        """

        pass

    def sample(self, mu: Tensor, var: Tensor):
        """Sample the z.

        Args:
            mu: the mu.
            var: the var.

        Returns:
            p: the p distribution.
            q: the q distribution.
            z: the z.
        """

        # Calculate the p and q
        std = torch.exp(var / 2)
        p = torch.distributions.Normal(torch.zeros_like(mu), torch.ones_like(std))
        q = torch.distributions.Normal(mu, std)

        # Sample the z
        z = q.rsample()

        # Return the output data
        return p, q, z

    def x_to_z_encoder(self, x: Tensor, input_index: int):
        """Encoder from x to z.

        Args:
            x: the input data.
            input_index: the index of the input data.

        Returns:
            z: the z.
        """

        # Encoder with index
        h = getattr(self, f"encoder_{input_index}")(x)
        # mu and var with index
        mu = getattr(self, f"fc_mu_{input_index}")(h)
        var = getattr(self, f"fc_var_{input_index}")(h)
        # Sample
        _, _, z = self.sample(mu, var)

        # Return z
        return z

    def z_to_x_decoder(self, z: Tensor, output_index: int):
        """Decoder from z to x.

        Args:
            z: the input data.
            output_index: the index of the output data.

        Returns:
            x_hat: the output data.
        """

        # Decoder with index
        x_hat = getattr(self, f"decoder_{output_index}")(z)

        # Return x_hat
        return x_hat

    def x_to_z_list_encoder(self, x_list: List[Tensor]):
        """Predict the z code.

        Args:
            x_list: the input data.

        Returns:
            z_list: the z code.
        """

        # Create the list to store the output data
        z_list = []

        # Predict the z code
        for i in range(len(self.x_dim)):
            # Encoder
            z = self.x_to_z_encoder(x_list[i], i)
            # Store the data
            z_list.append(z)

        # Return the output data
        return z_list

    def forward(self, x_list: List[Tensor]):
        """Forward the model.

        Args:
            x_list: the input data.

        Returns:
            x_hat_list: the output data.
        """

        # Create the list to store the output data
        x_hat_list = []

        # Forward the model
        for i in range(len(self.x_dim)):
            # Encoder
            z = self.x_to_z_encoder(x_list[i], i)
            # Decoder
            x_hat = self.z_to_x_decoder(z, i)
            x_hat_list.append(x_hat)

        # Return the output data
        return x_hat_list

    def forward_with_index(self, x: Tensor, input_index: int, output_index: int):
        """Forward the model with index.

        Args:
            x: the input data.
            input_index: the index of the input data.
            output_index: the index of the output data.

        Returns:
            x_hat: the output data.
        """

        # Encoder with index
        z = self.x_to_z_encoder(x, input_index)
        # Decoder with index
        x_hat = self.z_to_x_decoder(z, output_index)

        # Return the output data
        return x_hat

    def _prepare_batch(self, batch: Any):
        """Prepare the batch.

        Args:
            batch: the input batch.

        Returns:
            x: the input batch.
        """

        # Reshape the batch
        x = batch
        return x.view(x.size(0), -1)

    def _run_step(self, x_list):
        """Run the step.

        Args:
            x_list: the input data.

        Returns:
            x_hat_list: the output data.
            p_list: the p distribution.
            q_list: the q distribution.
            z_list: the z distribution.
            mu_list: the mu distribution.
            var_list: the var distribution.
        """

        # Create the list to store the output data
        x_hat_list = []
        p_list = []
        q_list = []
        z_list = []
        mu_list = []
        var_list = []

        # Run the step
        for i in range(len(self.x_dim)):
            # Encoder
            h = getattr(self, f"encoder_{i}")(x_list[i])
            # mu and var
            mu = getattr(self, f"fc_mu_{i}")(h)
            var = getattr(self, f"fc_var_{i}")(h)
            # Sample
            p, q, z = self.sample(mu, var)
            # Decoder
            x_hat = getattr(self, f"decoder_{i}")(z)
            # Store the data
            x_hat_list.append(x_hat)
            p_list.append(p)
            q_list.append(q)
            z_list.append(z)
            mu_list.append(mu)
            var_list.append(var)

        # Return the output data
        return x_hat_list, p_list, q_list, z_list, mu_list, var_list

    def step(self, batch: Any, batch_idx: int):
        """Step the model.

        Args:
            batch: the input batch.
            batch_idx: the batch index.

        Returns:
            loss: the loss.
            logs: the logs.
        """

        # Prepare the batch
        x = self._prepare_batch(batch)

        # Split the data
        x_list = []
        x_start = 0
        x_end = 0
        for i in range(len(self.x_dim)):
            x_end += self.x_dim[i]
            x_i = x[:, x_start:x_end]
            x_list.append(x_i)
            x_start = x_end

        # Run the step
        x_hat_list, p_list, q_list, z_list, _, _ = self._run_step(x_list)

        # Define the logs
        logs = {}

        # Calculate the reconstruction loss
        recon_loss_list = []
        for i in range(len(self.x_dim)):
            # MSE loss
            recon_loss_i = F.mse_loss(x_hat_list[i], x_list[i], reduction="mean")
            # Store the data
            logs[f"recon_loss_{i}"] = recon_loss_i
            recon_loss_list.append(recon_loss_i)
        # Average the loss
        recon_loss = sum(recon_loss_list) / len(recon_loss_list)
        # Store the data
        logs["recon_loss"] = recon_loss

        # Calculate the prediction loss
        pred_loss_list = []
        for i in range(len(self.x_dim) - 1):
            # Predict the x
            x_pred = self.forward_with_index(x_list[0], 0, i + 1)
            # MSE loss
            pred_loss_0_i = F.mse_loss(x_pred, x_list[i + 1], reduction="mean")
            # Store the data
            logs[f"pred_loss_0_{i + 1}"] = pred_loss_0_i
            pred_loss_list.append(pred_loss_0_i)
        # Average the loss
        pred_loss = sum(pred_loss_list) / len(pred_loss_list)
        # Store the data
        logs["pred_loss"] = pred_loss

        # Calculate the KL divergence loss
        kl_list = []
        for i in range(len(self.x_dim)):
            # KL divergence
            kl_i = (
                q_list[i].log_prob(z_list[i]) - p_list[i].log_prob(z_list[i])
            ).mean()
            # Store the data
            logs[f"kl_{i}"] = kl_i
            kl_list.append(kl_i)
        # Average the loss
        kl = sum(kl_list) / len(kl_list)
        # Store the data
        logs["kl"] = kl

        # Calculate the z loss
        z_loss_list = []
        for i in range(len(self.x_dim) - 1):
            for j in range(i + 1, len(self.x_dim)):
                # MSE loss
                z_loss_list.append(F.mse_loss(z_list[i], z_list[j], reduction="mean"))
        # Average the loss
        z_loss = sum(z_loss_list) / len(z_loss_list)
        # Store the data
        logs["z_loss"] = z_loss

        # Calculate the loss
        loss = (
            self.recon_pred_scale / (1 + self.recon_pred_scale) * recon_loss
            + 1 / (1 + self.recon_pred_scale) * pred_loss
            + self.kl_coef * kl
            + self.z_coef * z_loss
        )
        # Store the data
        logs["loss"] = loss

        # Return the loss and logs
        return loss, logs

    def training_step(self, batch: Any, batch_idx: int):
        """Training step.

        Args:
            batch: the input batch.
            batch_idx: the batch index.

        Returns:
            loss: the loss.
        """

        # Run the step
        loss, logs = self.step(batch, batch_idx)
        # Log the data
        self.log_dict(
            {f"train_{k}": v for k, v in logs.items()}, on_step=False, on_epoch=True
        )

        # Return the loss
        return loss

    def validation_step(self, batch: Any, batch_idx: int):
        """Validation step.

        Args:
            batch: the input batch.
            batch_idx: the batch index.

        Returns:
            loss: the loss.
        """

        # Run the step
        loss, logs = self.step(batch, batch_idx)
        # Log the data
        self.log_dict(
            {f"val_{k}": v for k, v in logs.items()}, on_step=False, on_epoch=True
        )

        # Return the loss
        return loss

    def test_step(self, batch: Any, batch_idx: int):
        """Test step.

        Args:
            batch: the input batch.
            batch_idx: the batch index.

        Returns:
            loss: the loss.
        """

        pass

    def test_epoch_end(self, outputs: List[Any]):
        """Test epoch end.

        Args:
            outputs: the outputs.

        Returns:
            None
        """

        pass

    def configure_optimizers(self):
        """Configure the optimizers.

        Returns:
            optimizer: the optimizer.
        """

        # Return the optimizer
        return torch.optim.Adam(self.parameters(), lr=self.lr)


if __name__ == "__main__":
    # Create the model
    model = MVAE()
    # Print the model
    print(model)
