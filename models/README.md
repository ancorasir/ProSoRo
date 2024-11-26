# Models

## Overview

This directory contains the models that are used in the project. The structure of the directory is as follows:

```plaintext
models/
├── README.md
├── mvae.py                         # MVAE model
├── data_module.py                  # Data module for PyTorch Lightning
└── checkpoints/                    # Trained models
    └── ...
```

## MVAE Model

The Multimodal Variational Autoencoder (MVAE) model is used to learn the latent space of the simulation data, including the poses, forces, and 3D shapes. The model is implemented in `mvae.py`. The model consists of three encoders and three decoders, each of which is responsible for encoding and decoding the poses, forces, and 3D shapes, respectively. The encoders and decoders are implemented as multi-layer perceptrons (MLPs) with two hidden layers. The latent space is a series of Gaussian distributions. Here are the hyperparameters of the MVAE model:

- `recon_pred_scale`: the scale of reconstruction and prediction loss.
- `z_coeff`: the coefficient of the MSE loss of the latent variable.
- `kl_coeff`: the coefficient of the KL divergence loss.
- `x_dim_list`: the dimension list of input data x.
- `h1_dim_list`: the dimension list of the first hidden layer.
- `h2_dim_list`: the dimension list of the second hidden layer.
- `z_dim`: the dimension of the latent variable.
- `lr`: the learning rate of the optimizer.

## Data Module

The data module is used to load the training data for PyTorch Lightning. The data module is implemented in `data_module.py`. Before training the MVAE model, the training data should be preprocessed and the normalizing parameters are saved.

## Trained Models

The trained MVAE models are stored in the `checkpoints/` directory. The models are saved as `mvae_{module_type}_{recon_pred_scale}_{z_coeff}_{kl_coeff}_{z_dim}.pth` and can be loaded using `torch.load()` function.

## Training and testing

Here we provide a [guide](../guide.ipynb) to train and test the MVAE model. You can run the notebook in your local environment.
