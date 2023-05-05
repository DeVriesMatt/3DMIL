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
        full_path = Path(self.h5_path) / f"{slide_id}.h5"
        hdf = pd.read_hdf(full_path)

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
        plate_num = "Plate" + str(self.new_df.loc[idx, "PlateNumber"])

        cell = tfl.imread(
            os.path.join(
                self.img_dir,
                plate_num,
                "stacked_intensity_cell",
                self.new_df.loc[idx, "serialNumber"] + ".tif",
            )
        )
        cell = cv2.normalize(cell, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
        nuc = tfl.imread(
            os.path.join(
                self.img_dir,
                plate_num,
                "stacked_intensity_nucleus",
                self.new_df.loc[idx, "serialNumber"] + ".tif",
            )
        )
        nuc = cv2.normalize(nuc, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
        erk = tfl.imread(
            os.path.join(
                self.img_dir,
                plate_num,
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
