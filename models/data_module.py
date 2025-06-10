#!/usr/bin/env python

from typing import Optional, Tuple
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, random_split
from pytorch_lightning import LightningDataModule


class ProSoRo(Dataset):
    def __init__(self, data: np.ndarray, transform=None):
        """Soft module dataset.

        Args:
            npy_file (string): Path to the npy file with annotations.
            transform (callable, optional): Optional transform to be applied on a sample.
        """

        self.data = data
        self.transform = transform

    def __len__(self):
        """Get the length of the dataset.

        Returns:
            len: int, the length of the dataset
        """

        return len(self.data)

    def __getitem__(self, idx):
        """Get the data item at the index.

        Args:
            idx: int, the index of the data item

        Returns:
            data_tensor: torch.Tensor, the data item
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()

        data_item = self.data[idx, :].astype("float32").reshape(-1, self.data.shape[1])
        data_tensor = torch.from_numpy(data_item)
        return data_tensor


class DataModule(LightningDataModule):
    def __init__(
        self,
        object: str = "cylinder",
        train_val_split: Tuple[float, float] = (0.875, 0.125),
        batch_size: int = 128,
        num_workers: int = 4,
        pin_memory: bool = False,
    ):
        """Data module for the soft module dataset.

        Args:
            object: str, the object name
            node_type: str, the type of the node
            data_size: int, the size of the data
            train_val_split: tuple, the split ratio of the training and validation data
            batch_size: int, the batch size
            num_workers: int, the number of workers
            pin_memory: bool, whether to pin the memory

        Returns:
            None
        """

        super().__init__()

        self.object = object
        self.train_val_split = train_val_split
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.data_min = None
        self.data_max = None

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

    def prepare_data(self):
        """
        Download data if needed. This method is called only from a single GPU.
        Do not use it to assign state (self.x = y).
        """
        pass

    def setup(self, stage: Optional[str] = None):
        """
        Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.
        This method is called by lightning separately when using `trainer.fit()` and `trainer.test()`!
        The `stage` can be used to differentiate whether the `setup()` is called before trainer.fit()` or `trainer.test()`.
        """
        if not self.data_train or not self.data_val or not self.data_test:
            source_path = "data/" + self.object + "/"
            file_path = source_path + "/training_data.npy"

            # normalize the data
            data = np.load(file_path)
            self.data_mu = np.mean(data, axis=0)
            self.data_std = np.std(data, axis=0)
            np.save(source_path + "/mu.npy", self.data_mu)
            np.save(source_path + "/std.npy", self.data_std)
            data = (data - self.data_mu) / self.data_std

            dataset = ProSoRo(data=data, transform=None)

            train_length = int(self.train_val_split[0] * len(dataset.data))
            val_length = int(len(dataset.data) - train_length)

            self.data_train, self.data_val = random_split(
                dataset,
                (train_length, val_length),
                generator=torch.Generator().manual_seed(42),
            )

    def train_dataloader(self):
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=True,
            shuffle=True,
            drop_last=True,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=True,
            shuffle=False,
            drop_last=True,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=True,
            shuffle=False,
            drop_last=True,
        )


if __name__ == "__main__":
    data_module = DataModule(
        object="cylinder",
        node_type="surface",
        train_val_split=(0.875, 0.125),
        batch_size=128,
        num_workers=4,
    )
    data_module.setup()

    x = data_module.train_dataloader().dataset[0]
    print(x.dtype)
