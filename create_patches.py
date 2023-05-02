# Description: This file contains the functions to create patch es from the
import torch
import torch.nn as nn
from torchvision.models import resnet18, resnet50
from acsconv.converters import ACSConverter
import numpy as np
import tifffile as tl
import cv2
from tqdm import tqdm
import pandas as pd
from tiler import Tiler
from utils import create_dir_if_not_exist


def create_model(resnet_size=50):
    """
    Creates a 3D model from a 2D model.
    :param resnet_size:
    :return:
    """
    if resnet_size == 18:
        model_2d = resnet18(pretrained=True)
    if resnet_size == 50:
        model_2d = resnet50(pretrained=True)

    model_3d = ACSConverter(model_2d)
    model_feat = nn.Sequential(*list(model_3d.model.children())[:-1])
    return model_feat


def create_patches(path, model, patch_size, overlap, save_path, num_plates=3):
    p = path
    for plate in range(1, num_plates + 1):
        new_p = p / f"Plate{plate}/run_0001/"

        for field in tqdm(sorted(new_p.iterdir())):
            if "DS_" in field.name:
                continue

            i = 0
            for folder in sorted(field.iterdir()):
                if "DS_" in folder.name:
                    continue

                if "scatter" not in folder.name:
                    imgs = []
                    for file in sorted(folder.iterdir()):
                        if "DS_" in file.name:
                            continue

                        if "tif" not in file.name:
                            continue
                        imgs.append(tl.imread(file))

                    if i == 0:
                        ch0 = np.asarray(imgs)
                        norm0 = cv2.normalize(
                            ch0, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX
                        )

                    if i == 1:
                        ch1 = np.asarray(imgs)
                        norm1 = cv2.normalize(
                            ch1, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX
                        )
                    if i == 2:
                        ch2 = np.asarray(imgs)
                        norm2 = cv2.normalize(
                            ch2, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX
                        )

                    i += 1

            allch = np.stack((norm0, norm1, norm2))

            tiler = Tiler(
                data_shape=allch.shape,
                tile_shape=(3, patch_size, patch_size, patch_size),
                channel_dimension=0,
                overlap=overlap,
            )
            tiles_useful = []
            for tile_id, tile in tiler.iterate(allch):
                if (tile[0]).sum() <= 50000:
                    continue
                else:
                    print(f"Tile {tile_id} out of {len(tiler)} tiles.")
                    tiles_useful.append(tile_id)

            tile_feats = []
            bbox_s = []
            for t in tiles_useful:
                tile_tensor = (
                    torch.from_numpy(tiler.get_tile(allch, t).astype(np.int16))
                    .unsqueeze(0)
                    .type(torch.FloatTensor)
                )
                t_feats = model(tile_tensor)
                tile_feats.append(t_feats.squeeze().detach().cpu().numpy())
                bbox_s.append(tiler.get_tile_bbox(t))

            save_dir = f"{save_path}/Plate{plate}/"
            create_dir_if_not_exist(save_dir)
            d = pd.DataFrame(tile_feats)
            d["bbox00"] = [b[0][0] for b in bbox_s]
            d["bbox01"] = [b[0][1] for b in bbox_s]
            d["bbox02"] = [b[0][2] for b in bbox_s]
            d["bbox10"] = [b[1][0] for b in bbox_s]
            d["bbox11"] = [b[1][1] for b in bbox_s]
            d["bbox12"] = [b[1][2] for b in bbox_s]
            d.to_hdf(f"{save_dir}/{field.name}.h5", key="da", mode="w")
