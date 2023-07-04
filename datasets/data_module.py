import pytorch_lightning as pl
import torch.utils.data

from .dataset import FieldData
from .dataset import CellData
from torch.utils.data import DataLoader
import numpy as np


class SingleCellDataModule(pl.LightningDataModule):
    def __init__(
        self,
        h5_path=None,
        csv_path=None,
        state=None,
        shuffle=True,
        batch_size=1,
        drug_label="Blebbistatin",
        task="PredictDrug",
    ):
        super().__init__()
        # Set all input args as attributes
        self.__dict__.update(locals())
        self.h5_path = h5_path
        self.csv_path = csv_path
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.drug_label = drug_label
        self.task = task

    def setup(self, stage=None):
        if self.task == "PredictDrug":
            self.train_dset = CellData(
                h5_path=self.h5_path,
                csv_path=self.csv_path,
                state="train",
                drug_label=self.drug_label,
            )
            self.val_dset = CellData(
                h5_path=self.h5_path,
                csv_path=self.csv_path,
                state="val",
                drug_label=self.drug_label,
            )
            self.test_dset = CellData(
                h5_path=self.h5_path,
                csv_path=self.csv_path,
                state="test",
                drug_label=self.drug_label,
            )

        else:
            self.train_dset = FieldData(
                h5_path=self.h5_path, csv_path=self.csv_path, state="train"
            )

            self.val_dset = FieldData(
                h5_path=self.h5_path, csv_path=self.csv_path, state="val"
            )

            self.test_dset = FieldData(
                h5_path=self.h5_path, csv_path=self.csv_path, state="test"
            )

    def calculate_weights(self):
        dloader = DataLoader(self.train_dset, batch_size=1, shuffle=False)
        labels = []
        for d in dloader:
            labels.append(d[1].item())
        labels = np.asarray(labels)
        class_counts = np.bincount(labels)

        # Calculate the inverse of each class frequency
        class_weights = 1.0 / class_counts

        # Now, you can create a weight for each instance in the dataset
        weights = class_weights[labels]
        return torch.from_numpy(weights)

    def train_dataloader(self):
        return DataLoader(
            self.train_dset,
            batch_size=self.batch_size,
            sampler=torch.utils.data.WeightedRandomSampler(
                weights=self.calculate_weights(), num_samples=73
            ),
        )

    def val_dataloader(self):
        return DataLoader(self.val_dset, batch_size=self.batch_size, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.test_dset, batch_size=self.batch_size, shuffle=False)
