#!/usr/bin/env python

import os
import shutil
import torch
import pytorch_lightning as L
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from models.data_module import DataModule
from models.mvae import MVAE

# Set the training function
def train(module_type: str='cylinder',
          recon_pred_scale: float=1,
          z_coeff: float=1,
          kl_coeff: float=0.1,
          x_dim_list: list=[6, 6, 1749],
          h1_dim_list: list=[16, 16, 1024],
          h2_dim_list: list=[32, 32, 256],
          z_dim: int=64,
          lr: float=1e-5,
          batch_size: int=128,
          max_epochs: int=500) -> None:
    ''' Train the MVAE model.

    In this function, the MVAE model is trained with the given parameters.
    The trained model is saved in the './models/pths' folder.
    The log files are saved in the './lightning_logs' folder.

    Args:
        module_type: 'cylinder', 'octagonalPrism', 'quadraticPrism', 'neck', 'origami'
        node_type: 'surface', 'centralLine'
        recon_pred_scale: coresponds to alpha in the paper
        z_coeff: coresponds to zeta in the paper
        kl_coeff: coresponds to gamma in the paper
        x_dim_list: the dimension of the input data
        z_dim: the dimension of the latent space
        lr: the learning rate
        batch_size: the batch size
        max_epochs: the maximum number of epochs

    Returns:
        None
    '''

    ## DataModule
    dm = DataModule(module_type=module_type, 
                    batch_size=batch_size, 
                    num_workers=8)
    dm.setup()

    ## Callbacks
    trainer = L.Trainer(max_epochs=max_epochs, 
                        accelerator="cuda", 
                        devices=1, 
                        strategy="auto", 
                        callbacks=[ModelCheckpoint(save_weights_only=True),
                                   LearningRateMonitor("epoch")])

    ## Instantiate
    model = MVAE(x_dim_list=x_dim_list,
                 h1_dim_list=h1_dim_list,
                 h2_dim_list=h2_dim_list,
                 z_dim=z_dim, 
                 lr=lr,
                 recon_pred_scale=recon_pred_scale,
                 z_coeff=z_coeff,
                 kl_coeff=kl_coeff)

    ## Training
    trainer.fit(model, dm)

    ## Evaluation
    model.eval()

    ## Save the model
    if not os.path.exists('models/pths'):
        os.makedirs('models/pths')

    torch.save(model.state_dict(), 
                'models/pths/mvae_' + \
                module_type + '_' + \
                str(recon_pred_scale) + '_' + \
                str(z_coeff) + '_' + \
                str(kl_coeff) + '_' + \
                str(z_dim) + '.pth')

    ## Rename log folder
    version_folder = 'lightning_logs/version_' + str(trainer.logger.version)
    new_version_folder = './lightning_logs/mvae_' + \
                            module_type + '_' + \
                            str(recon_pred_scale) + '_' + \
                            str(z_coeff) + '_' + \
                            str(kl_coeff) + '_' + \
                            str(z_dim)
    if os.path.exists(new_version_folder):
        shutil.rmtree(new_version_folder)
    os.rename(version_folder, new_version_folder)

if __name__ == "__main__":
    # module_type: module type
    module_type = 'cylinder'
    # loss = alpha / (1 + alpha) * recon_loss + 1/ (1 + alpha) * pred_loss + zeta * z_loss + gamma * kl_loss
    # recon_pred_scale: coresponds to alpha in the paper
    # recon_coeff: recon_pred_scale / (1 + recon_pred_scale)
    # pred_coeff: 1 / (1 + recon_pred_scale)
    recon_pred_scale = 1
    # z_coeff: coresponds to zeta in the paper
    z_coeff = 1
    # kl_coeff: coresponds to gamma in the paper
    kl_coeff = 0.1
    # x_dim_list: the dimension of the input data
    x_dim_list = [6, 6, 2736]
    # h1_dim_list: the dimension of the hidden layer 1
    h1_dim_list = [16, 16, 1024]
    # h2_dim_list: the dimentsion of the hidden layer 2
    h2_dim_list = [32, 32, 256]
    # z_dim: the dimension of the latent space
    z_dim = 32
    # lr: the learning rate
    lr = 1e-5
    # batch_size: the batch size
    batch_size = 128
    # max_epochs: the maximum number of epochs
    max_epochs = 1000

    train(module_type=module_type,
          recon_pred_scale=recon_pred_scale,
          z_coeff=z_coeff,
          kl_coeff=kl_coeff,
          x_dim_list=x_dim_list,
          h1_dim_list=h1_dim_list,
          h2_dim_list=h2_dim_list,
          z_dim=z_dim,
          lr=lr,
          batch_size=batch_size,
          max_epochs=max_epochs)