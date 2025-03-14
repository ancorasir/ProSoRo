# Data

## Overview

This directory contains the data that are used in the project. The structure of the directory is as follows:

```plaintext
data/
├── README.md
├── module_type/
│   ├── pose.csv                      # Pose data
│   ├── data.npy                      # Data
│   ├── training_data.npy             # Training data
│   ├── testing_data.npy              # Testing data
│   └── ...
└── ...
```

## Data Example

Data for each module are stored in a separate directory named by the module type, and the data files are stored in the directory. The data files include:

- `pose.csv`: Pose data that contains the boundary conditions of the simulation.
- `data.npy`: Raw data that are generated from the simulation.
- `training_data.npy`: Training data that are used to train the model.
- `testing_data.npy`: Testing data that are used to evaluate the model.

## Data Generation

Please follow the [guide](../guide.ipynb) to generate the data. You can also download the pre-generated data from the [dataset](https://drive.google.com/drive/folders/1DLtiRO6jbu13QDrZpORM2CHrvn-JiiRk?usp=sharing).
