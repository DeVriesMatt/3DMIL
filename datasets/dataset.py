import pandas as pd
import numpy as np
import cv2
import os
import torch
from torch.utils.data import Dataset
from sklearn import preprocessing
import random
from pathlib import Path
import tifffile as tfl


class GEFGAPData(Dataset):
    """
    Dataset class for the GEFGAP dataset
    """

    def __init__(self, h5_path=None, csv_path=None, state=None, shuffle=False):
        # Set all input args as attributes
        self.__dict__.update(locals())
        self.h5_path = h5_path
        self.csv_path = csv_path
        self.shuffle = shuffle

        self.slide_data = pd.read_csv(self.csv_path)

        self.state = state

        self.pwn = np.unique(
            [
                f"{self.slide_data.loc[i, 'PlateNumber']}"
                f"{self.slide_data.loc[i, 'well_nuc']}"
                for i in range(len(self.slide_data))
            ]
        )

    def __len__(self):
        return len(self.pwn)

    def __getitem__(self, idx):
        slide_id = self.pwn[idx]
        full_path = Path(self.h5_path) / f"{slide_id}.csv"
        hdf = pd.read_csv(full_path)

        features = torch.from_numpy(hdf.iloc[:, :100].values).type(torch.DoubleTensor)
        label = np.unique(hdf["Treatment"].values)
        treatment = np.unique(hdf["Treatment"].values)
        serial_number = hdf["serialNumber"].values

        # ----> shuffle
        if self.shuffle:
            index = [x for x in range(features.shape[0])]
            random.shuffle(index)
            features = features[index]

        return features, label, slide_id, treatment, serial_number.tolist()


class CellData(Dataset):
    """
    Dataset class for the single cell dataset
    """

    def __init__(
        self,
        h5_path=None,
        csv_path=None,
        state=None,
        shuffle=False,
        drug_label="Blebbistatin",
    ):
        # Set all input args as attributes
        self.__dict__.update(locals())
        self.h5_path = h5_path
        self.csv_path = csv_path
        self.shuffle = shuffle
        self.drug_label = drug_label

        self.slide_data = pd.read_csv(self.csv_path)
        labels = {
            "Binimetinib": 0,
            "Blebbistatin": 0,
            "CK666": 0,
            "DMSO": 0,
            "H1152": 0,
            "MK1775": 0,
            "No Treatment": 0,
            "Nocodazole": 0,
            "PF228": 0,
            "Palbociclib": 0,
        }
        self.label_dict = {k: 1 if k == self.drug_label else 0 for k in labels}

        # ---->split dataset
        self.state = state
        self.pwn = np.unique(
            self.slide_data[
                (self.slide_data["Splits"] == self.state)  # &
                #             (self.slide_data['Class']!='Trials') &
                #             (self.slide_data['Class']!='Successful')
            ]["PlateWellNum"]
            .reset_index(drop=True)
            .values
        )
        self.label = self.slide_data[
            (self.slide_data["Splits"] == self.state)  # &
            #             (self.slide_data['Class']!='Trials') &
            #             (self.slide_data['Class']!='Successful')
        ]["Treatment"].reset_index(drop=True)

    def __len__(self):
        return len(self.pwn)

    def __getitem__(self, idx):
        slide_id = self.pwn[idx]
        full_path = Path(self.h5_path) / f"{slide_id}.csv"
        hdf = pd.read_csv(full_path)

        features = torch.from_numpy(hdf.iloc[:, :100].values).type(torch.DoubleTensor)
        label = torch.tensor(int(self.label_dict[np.unique(hdf["Treatment"])[0]]))
        treatment = np.unique(hdf["Treatment"])[0]
        serial_number = hdf["serialNumber"].values

        # ----> shuffle
        if self.shuffle:
            index = [x for x in range(features.shape[0])]
            random.shuffle(index)
            features = features[index]

        return features, label, slide_id, treatment, serial_number.tolist()


class FieldData(Dataset):
    def __init__(self, h5_path=None, csv_path=None, state=None, shuffle=False):
        super(FieldData, self).__init__()
        # Set all input args as attributes
        self.__dict__.update(locals())
        self.h5_path = h5_path
        self.csv_path = csv_path
        self.shuffle = shuffle

        self.slide_data = pd.read_csv(self.csv_path)
        self.label_dic = {
            "CellToxic": 1,
            "Control": 0,
            "NotUseful": 1,
            "Successful": 2,
            "Trials": 2,
        }

        # ---->split dataset
        self.state = state
        self.pwn = np.unique(
            self.slide_data[
                (self.slide_data["Splits"] == self.state)
                & (self.slide_data["Class"] != "Trials")
            ]["fieldNumber"]
            .reset_index(drop=True)
            .values
        )
        self.label = self.slide_data[
            (self.slide_data["Splits"] == self.state)
            & (self.slide_data["Class"] != "Trials")
        ]["Class"].reset_index(drop=True)

        self.drug = self.slide_data[
            (self.slide_data["Splits"] == self.state)
            & (self.slide_data["Class"] != "Trials")
        ]["Treatment"].reset_index(drop=True)

    def __len__(self):
        return len(self.pwn)

    def __getitem__(self, idx):
        slide_id = f"field_{str(self.pwn[idx]).zfill(4)}"
        full_path = Path(self.h5_path) / f"{slide_id}.csv"
        hdf = pd.read_csv(full_path)

        features = torch.from_numpy(hdf.iloc[:, :2048].values).type(torch.DoubleTensor)
        label = torch.tensor(
            int(
                self.label_dic[
                    np.unique(
                        self.slide_data[self.slide_data["fieldNumber"] == slide_id][
                            "Class"
                        ]
                    )[0]
                ]
            )
        )
        treatment = self.drug[idx].values[0]

        # ----> shuffle
        if self.shuffle:
            index = [x for x in range(features.shape[0])]
            random.shuffle(index)
            features = features[index]

        return features, label, slide_id, treatment


class VoxData(Dataset):
    def __init__(
        self, annotations_file, img_dir, img_size=400, transform=None, partition="all"
    ):
        self.annot_df = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.img_size = img_size
        self.transform = transform
        self.partition = partition
        if self.partition != "all":
            self.new_df = self.annot_df[
                (self.annot_df.xDim <= self.img_size)
                & (self.annot_df.yDim <= self.img_size)
                & (self.annot_df.zDim <= self.img_size)
                & (self.annot_df.xDim >= 11)
                & (self.annot_df.yDim >= 11)
                & (self.annot_df.zDim >= 11)
                & (self.annot_df.Splits == self.partition)
                & (self.annot_df.PlateNumber == 1)
            ].reset_index(drop=True)
        else:
            self.new_df = self.annot_df[
                (self.annot_df.xDim <= self.img_size)
                & (self.annot_df.yDim <= self.img_size)
                & (self.annot_df.zDim <= self.img_size)
                & (self.annot_df.xDim >= 11)
                & (self.annot_df.yDim >= 11)
                & (self.annot_df.zDim >= 11)
                & (self.annot_df.PlateNumber == 1)
            ].reset_index(drop=True)

        self.le = preprocessing.LabelEncoder()
        self.le.fit(self.new_df["Treatment"].values)

    def __len__(self):
        return len(self.new_df)

    def __getitem__(self, idx):
        # read the image
        treatment = self.new_df.loc[idx, "Treatment"]

        cell = tfl.imread(
            os.path.join(
                self.img_dir,
                "stacked_intensity_cell",
                self.new_df.loc[idx, "serialNumber"] + ".tif",
            )
        )
        cell = cv2.normalize(cell, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
        nuc = tfl.imread(
            os.path.join(
                self.img_dir,
                "stacked_intensity_nucleus",
                self.new_df.loc[idx, "serialNumber"] + ".tif",
            )
        )
        nuc = cv2.normalize(nuc, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
        erk = tfl.imread(
            os.path.join(
                self.img_dir,
                "stacked_erk",
                self.new_df.loc[idx, "serialNumber"] + ".tif",
            )
        )
        erk = cv2.normalize(erk, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)

        image = np.stack((cell, erk, nuc))
        image = torch.from_numpy(image.astype(np.int16))
        if self.transform is not None:
            image = self.transform(image)

        serial_number = self.new_df.loc[idx, "serialNumber"]
        enc_labels = torch.tensor(self.le.transform([treatment]))
        return image, enc_labels, treatment, serial_number
