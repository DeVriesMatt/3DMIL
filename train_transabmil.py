import os
import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.loggers.neptune import NeptuneLogger
import argparse
import warnings
from datasets.data_module import SingleCellDataModule
from models.transabmil import TransABMIL

warnings.filterwarnings("ignore")


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def get_args():
    parser = argparse.ArgumentParser(description="Train TransABMIL model for all drugs")

    parser.add_argument(
        "--csv_dir",
        type=str,
        default="/mnt/nvme0n1/Datasets/SingleCellFromNathan_17122021"
        "/all_data_removedwrong_ori_removedTwo_train_test.csv",
        help="Path to CSV file containing data paths",
    )
    parser.add_argument(
        "--img_dir",
        type=str,
        default="/mnt/nvme0n1/Datasets/TransformerFeats/",
        help="Path to directory containing images",
    )
    parser.add_argument(
        "--batch_size", type=int, default=1, help="Batch size for training"
    )
    parser.add_argument(
        "--max_epochs", type=int, default=500, help="Maximum number of training epochs"
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=os.cpu_count(),
        help="Number of data loading workers",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="logs",
        help="Directory to save the model weights to",
    )
    parser.add_argument(
        "--num_gpus",
        type=int,
        default=1,
        help="Number of GPUs to use for training",
    )
    parser.add_argument(
        "--lr", type=float, default=0.00001, help="Learning rate for optimizer"
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=1e-4,
        help="Weight decay parameter for optimizer",
    )
    parser.add_argument(
        "--input_size",
        type=int,
        default=128,
        help="Hidden dimension of projection head",
    )
    parser.add_argument(
        "--log_dir", type=str, default="./logs/", help="directory to save logs"
    )
    return parser.parse_args()


def train(args):
    # Setting the seed
    pl.seed_everything(42)
    early_stop_callback = EarlyStopping(
        monitor="val_loss", min_delta=0.00, patience=100, verbose=True, mode="min"
    )
    checkpoint_callback = ModelCheckpoint(monitor="val_loss")
    lr_monitor = LearningRateMonitor(logging_interval="step")

    cell_data = SingleCellDataModule(
        csv_path=args.csv_dir,
        h5_path=args.img_dir,
        batch_size=args.batch_size,
        task="PredictDrug",
        drug_label=args.drug_label,
    )
    cell_data.setup()
    model = TransABMIL(
        num_classes=2, criterion=nn.BCEWithLogitsLoss(weight=torch.tensor([9.0]).cuda())
    )
    neptune_logger = NeptuneLogger(
        project="mattdevries/TransABMIL",
        api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm"
        "5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly"
        "9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiIxMjd"
        "mMmFhZS0xN2Y1LTQ2MzEtOTc3OS03ZmY1MTVhNzU0NTkifQ==",
        name=args.drug_label,
        tags=[args.drug_label],
    )

    trainer = pl.Trainer(
        accelerator="gpu",
        devices=args.num_gpus,
        max_epochs=args.max_epochs,
        callbacks=[early_stop_callback, checkpoint_callback, lr_monitor],
        default_root_dir=args.log_dir,
        logger=neptune_logger,
    )
    trainer.fit(model, cell_data)
    trainer.test(model=model, datamodule=cell_data)


def main(args):
    drugs = [
        "Binimetinib",
        "Blebbistatin",
        "CK666",
        "DMSO",
        "H1152",
        "MK1775",
        "No Treatment",
        "Nocodazole",
        "PF228",
        "Palbociclib",
    ]
    for drug in drugs:
        print(f"Training model for drug: {drug}")
        setattr(args, "drug_label", drug)
        setattr(args, "log_dir", os.path.join(args.log_dir, drug))
        train(args)
        print(f"Finished training model for drug: {drug}")


if __name__ == "__main__":
    args = get_args()
    main(args)
