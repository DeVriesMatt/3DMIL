import torch
from torch import nn

import pytorch_lightning as pl
from torchmetrics import Accuracy, AUROC


class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.trans = nn.Transformer(
            d_model=100, nhead=10, num_encoder_layers=12, batch_first=True
        ).encoder
        self.mlp = nn.Linear(100, 3)

    def forward(self, x):
        trans_out = self.trans(x.float())
        out = self.mlp(trans_out)
        out = out.mean(axis=1)
        return out, trans_out


class CloudClassifierPL(pl.LightningModule):
    def __init__(self, criterion=nn.CrossEntropyLoss(), num_classes=3):
        super(CloudClassifierPL, self).__init__()

        self.save_hyperparameters(ignore=["criterion", "model"])
        self.lr = 0.00001
        self.criterion = criterion
        self.model = Classifier()
        self.num_classes = num_classes
        self.accuracy_macro = Accuracy(
            task="multiclass", num_classes=num_classes, average="macro"
        )
        self.accuracy_micro = Accuracy(
            task="multiclass", num_classes=num_classes, average="micro"
        )
        self.accuracy_weighted = Accuracy(
            task="multiclass", num_classes=num_classes, average="weighted"
        )
        self.AUC = AUROC(task="multiclass", num_classes=num_classes)

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.lr,
            betas=(0.9, 0.999),
            weight_decay=1e-6,
        )
        return optimizer

    def training_step(self, batch, batch_idx):
        inputs, labels = batch[0], batch[1]
        outputs, _ = self.model(inputs)
        loss = self.criterion(outputs, labels)
        preds = torch.softmax(outputs, dim=1).argmax()
        acc = self.accuracy_weighted(preds.unsqueeze(0), labels)
        auc = self.AUC(torch.softmax(outputs, dim=1), labels)

        self.log("train_loss", loss, on_step=True, on_epoch=True, logger=True)
        self.log(
            "train_acc",
            acc,
            on_step=True,
            on_epoch=True,
            logger=True,
            prog_bar=True,
        )
        self.log("train_auc", auc, on_step=True, on_epoch=True, logger=True)
        dic = {"loss": loss, "acc": acc, "auc": auc}
        return dic

    def validation_step(self, batch, batch_idx):
        inputs, labels = batch[0], batch[1]
        outputs, _ = self.model(inputs)

        loss = self.criterion(outputs, labels)
        preds = torch.softmax(outputs, dim=1).argmax()
        acc = self.accuracy_weighted(preds.unsqueeze(0), labels)
        self.log("val_loss", loss, on_step=True, on_epoch=True, logger=True)
        self.log("val_acc", acc, on_step=True, on_epoch=True, logger=True)

    def test_step(self, batch, batch_idx):
        inputs, labels = batch[0], batch[1]
        outputs, _ = self.model(inputs)
        loss = self.criterion(outputs, labels)
        preds = torch.softmax(outputs, dim=1).argmax()

        acc_macro = self.accuracy_macro(preds.unsqueeze(0), labels)
        acc_micro = self.accuracy_micro(preds.unsqueeze(0), labels)
        acc_weighted = self.accuracy_weighted(preds.unsqueeze(0), labels)
        auc = self.AUC(torch.softmax(outputs, dim=1), labels)
        self.log("test_loss", loss, on_step=True, on_epoch=True, logger=True)
        self.log(
            "test_acc_macro",
            acc_macro,
            on_step=True,
            on_epoch=True,
            logger=True,
        )
        self.log(
            "test_acc_micro",
            acc_micro,
            on_step=True,
            on_epoch=True,
            logger=True,
        )
        self.log(
            "test_acc_weighted",
            acc_weighted,
            on_step=True,
            on_epoch=True,
            logger=True,
        )
        self.log("test_auc", auc, on_step=True, on_epoch=True, logger=True)
