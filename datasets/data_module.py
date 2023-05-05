import lightning.pytorch as pl
from dataset import FieldData
from torch.utils.data import DataLoader


class SingleCellDataModule(pl.LightningDataModule):
    def __init__(
        self, h5_path=None, csv_path=None, state=None, shuffle=True, batch_size=1
    ):
        super().__init__()
        # Set all input args as attributes
        self.__dict__.update(locals())
        self.h5_path = h5_path
        self.csv_path = csv_path
        self.shuffle = shuffle
        self.batch_size = batch_size

    def setup(self, stage=None):
        self.train_dset = FieldData(
            h5_path=self.h5_path, csv_path=self.csv_path, state="train"
        )

        self.val_dset = FieldData(
            h5_path=self.h5_path, csv_path=self.csv_path, state="val"
        )

        self.test_dset = FieldData(
            h5_path=self.h5_path, csv_path=self.csv_path, state="test"
        )

    def train_dataloader(self):
        return DataLoader(self.train_dset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dset, batch_size=self.batch_size, shuffle=True)

    def test_dataloader(self):
        return DataLoader(self.test_dset, batch_size=self.batch_size, shuffle=False)
