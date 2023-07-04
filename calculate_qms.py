from train_transabmil_wandb import get_args
from models.transabmil import TransABMIL
import torch
from tqdm import tqdm
import os
import pandas as pd
from datasets.dataset import GEFGAPData
import torch.nn as nn


def calculate_qms():
    args = get_args()
    dataframe_dir = os.path.join(args.log_dir, args.drug_label)
    dset = GEFGAPData(h5_path=args.img_dir, csv_path=args.csv_dir, state="train")
    model_path = os.path.join(args.log_dir, args.drug_label, "TransABMIL")
    model_path = os.path.join(model_path, os.listdir(model_path)[0], "checkpoints")
    model_path = os.path.join(model_path, os.listdir(model_path)[0])

    model = TransABMIL(
        num_classes=2, criterion=nn.BCEWithLogitsLoss(weight=torch.tensor([9.0]).cuda())
    )
    model.load_state_dict(torch.load(model_path)["state_dict"])
    model.eval()

    y_true = []
    y_pred = []
    treats = []
    ids = []

    attns = []
    classes = []

    serials = []
    model.eval()

    for i, d in tqdm(enumerate(dset)):
        outputs = model.model(torch.squeeze(d[0]))

        predicts = torch.sigmoid(outputs[1][0])
        treats.append(d[3][0])
        ids.append(d[2])
        attns.append(outputs[4])
        classes.append(outputs[0])
        serials.append(d[-1])

        y_true.append(d[1][0])
        if i == 0:
            preds = torch.sigmoid(outputs[1][0]).detach()
            y_pred = predicts.unsqueeze(0).detach()
        else:
            preds = torch.cat((preds, torch.sigmoid(outputs[1][0]).detach()))
            y_pred = torch.cat((y_pred, predicts.unsqueeze(0).detach()))

    df = pd.DataFrame(y_true, columns=["GeneKnockdown"])
    df[args.drug_label] = y_pred.numpy()
    df["id"] = ids
    df.to_csv(os.path.join(dataframe_dir, "qms.csv"))

    return y_true, y_pred, treats, ids, attns, classes, serials


if __name__ == "__main__":
    calculate_qms()
