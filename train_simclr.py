import os
import torch.utils.data as data
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from models.simclr import SimCLR
from torchvision.models import resnet18, resnet50
from acsconv.converters import ACSConverter
from datasets import VoxData, ContrastiveTransformations, contrast_transforms
import argparse
import warnings

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
    parser = argparse.ArgumentParser(description="Train SimCLR model")

    parser.add_argument(
        "--resnet_size",
        type=int,
        default=18,
        choices=[18, 50],
        help="Size of the ResNet backbone",
    )
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
        default="/mnt/nvme0n1/Datasets/SingleCellFromNathan_17122021/",
        help="Path to directory containing images",
    )
    parser.add_argument(
        "--batch_size", type=int, default=8, help="Batch size for training"
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
        "--lr", type=float, default=5e-4, help="Learning rate for optimizer"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.07,
        help="Temperature parameter for contrastive loss function",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=1e-4,
        help="Weight decay parameter for optimizer",
    )
    parser.add_argument(
        "--hidden_dim",
        type=int,
        default=128,
        help="Hidden dimension of projection head",
    )
    parser.add_argument(
        "--n_views",
        type=int,
        default=2,
        help="Number of views for contrastive learning",
    )
    return parser.parse_args()


def main(args):
    # Setting the seed
    pl.seed_everything(42)
    if args.resnet_size == 18:
        model_2d = resnet18(pretrained=True)
    elif args.resnet_size == 50:
        model_2d = resnet50(pretrained=True)
    else:
        raise NotImplementedError
    # Convert the model to use ACSConv instead of regular convolutions
    model_3d = ACSConverter(model_2d)
    # Load the dataset
    dataset = VoxData(
        annotations_file=args.csv_dir,
        img_dir=args.img_dir,
        transform=ContrastiveTransformations(contrast_transforms, n_views=args.n_views),
    )
    # Train the model
    simclr_model = train_simclr(
        dataset=dataset,
        batch_size=args.batch_size,
        hidden_dim=args.hidden_dim,
        lr=args.lr,
        temperature=args.temperature,
        weight_decay=args.weight_decay,
        max_epochs=args.max_epochs,
        num_workers=args.num_workers,
        model_3d=model_3d.model,
        save_dir=args.save_dir,
        num_gpus=args.num_gpus,
    )
    return simclr_model


def train_simclr(
    batch_size,
    dataset,
    max_epochs=500,
    num_workers=os.cpu_count(),
    model_3d=None,
    save_dir=None,
    num_gpus=1,
    **kwargs,
):
    trainer = pl.Trainer(
        default_root_dir=os.path.join(save_dir, "logs"),
        accelerator="auto",
        devices=num_gpus,
        max_epochs=max_epochs,
        callbacks=[
            ModelCheckpoint(save_weights_only=True, mode="min", monitor="train_loss"),
            LearningRateMonitor("epoch"),
        ],
    )
    trainer.logger._default_hp_metric = (
        None  # Optional logging argument that we don't need
    )

    # Check whether pretrained model exists. If yes, load it and skip training
    pretrained_filename = os.path.join(save_dir, "logs", "default", "version_0")
    if os.path.isfile(pretrained_filename):
        print(f"Found pretrained model at {pretrained_filename}, loading...")
        # Automatically loads the model with the saved hyperparameters
        model = SimCLR.load_from_checkpoint(pretrained_filename)
    else:
        train_loader = data.DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=True,
            drop_last=True,
            pin_memory=True,
            num_workers=num_workers,
        )

        model = SimCLR(max_epochs=max_epochs, model_3d=model_3d, **kwargs)
        trainer.fit(model, train_loader)
        # Load the best checkpoint after training
        model = SimCLR.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)

    return model


if __name__ == "__main__":
    args = get_args()
    main(args)
