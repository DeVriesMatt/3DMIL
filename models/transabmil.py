import torch.nn.functional as F
import torch
from torch import nn

import pytorch_lightning as pl
from torchmetrics import Accuracy, AUROC, F1Score, Precision, Recall, MetricCollection
from datasets.data_augmentation import DataAugmentation
import pandas as pd


class TransIClassifier(nn.Module):
    def __init__(
        self, in_size=100, out_size=1, linear=False, dropout=0.2, fully_connected=True
    ):
        super(TransIClassifier, self).__init__()
        self.fully_connected = fully_connected
        self.trans = nn.Transformer(
            d_model=in_size,
            nhead=in_size // 10,
            num_encoder_layers=12,
            batch_first=True,
        ).encoder
        if linear:
            self.mlp = nn.Linear(in_size, out_size)
        else:
            self.mlp = nn.Sequential(
                nn.Linear(in_size, 1), nn.ReLU(), nn.Dropout(dropout)
            )

    def forward(self, x):
        trans_out = self.trans(x.float())
        if self.fully_connected:
            out = self.mlp(trans_out)
        else:
            out = trans_out.mean(axis=1)
        return trans_out, out


class FCLayer(nn.Module):
    def __init__(self, in_size=100, out_size=1):
        super(FCLayer, self).__init__()
        self.fc1 = nn.Sequential(nn.Linear(in_size, 512), nn.ReLU(), nn.Dropout(0.2))
        self.fc2 = nn.Sequential(nn.Linear(512, 128), nn.ReLU(), nn.Dropout(0.2))
        self.fc3 = nn.Sequential(nn.Linear(128, out_size), nn.ReLU(), nn.Dropout(0.2))

    def forward(self, feats):
        x = self.fc1(feats)
        x = self.fc2(x)
        x = self.fc3(x)
        return feats, x


class IClassifier(nn.Module):
    def __init__(self, feature_size=100, output_class=1, dropout=0.2):
        super(IClassifier, self).__init__()

        self.fc = nn.Sequential(
            nn.Linear(feature_size, output_class), nn.ReLU(), nn.Dropout(dropout)
        )

    def forward(self, x):
        c = self.fc(x.float())  # N x C
        return x.float(), c


class BClassifier(nn.Module):
    def __init__(
        self,
        input_size=100,
        output_class=1,
        dropout_v=0.2,
        nonlinear=True,
        passing_v=True,
    ):  # K, L, N
        super(BClassifier, self).__init__()
        if nonlinear:
            self.q = nn.Sequential(
                nn.Linear(input_size, 128),
                nn.ReLU(),
                nn.Linear(128, 128),
                nn.Tanh(),
                nn.Dropout(0.2),
            )
        else:
            self.q = nn.Linear(input_size, 128)
        if passing_v:
            self.v = nn.Sequential(
                nn.Dropout(dropout_v), nn.Linear(input_size, input_size), nn.ReLU()
            )
        else:
            self.v = nn.Identity()

        # 1D convolutional layer that can handle multiple class (including binary)
        self.fcc = nn.Conv1d(output_class, output_class, kernel_size=input_size)

    def forward(self, feats, c):  # N x K, N x C
        V = self.v(feats)  # N x V, unsorted
        Q = self.q(feats).view(feats.shape[0], -1)  # N x Q, unsorted

        # handle multiple classes without for loop
        _, m_indices = torch.sort(
            c, 0, descending=True
        )  # sort class scores along the instance dimension, m_indices in shape N x C
        #         print(m_indices[0, :])
        m_feats = torch.index_select(feats, dim=0, index=m_indices[0, :])
        # select critical instances, m_feats in shape C x K
        q_max = self.q(
            m_feats
        )  # compute queries of critical instances, q_max in shape C x Q
        #         print(Q.transpose(0, 1).shape)
        #         print(q_max.transpose(0, 1).shape)
        A_raw = torch.mm(
            Q, q_max.transpose(0, 1)
        )  # compute inner product of Q to each entry of q_max,
        #         A in shape N x C, each column contains unnormalized attention scores
        A = F.softmax(
            A_raw / torch.sqrt(torch.tensor(Q.shape[1], dtype=torch.float32)), 0
        )
        # normalize attention scores, A in shape N x C,

        B = torch.mm(
            A.transpose(0, 1), V
        )  # compute bag representation, B in shape C x V

        B = B.view(1, B.shape[0], B.shape[1])  # 1 x C x V
        C = self.fcc(B)  # 1 x C x 1
        C = C.view(1, -1)
        return C, A, B, A_raw


class MILNet(nn.Module):
    def __init__(
        self,
        i_class="trans",
    ):
        super(MILNet, self).__init__()
        if i_class == "trans":
            self.i_classifier = TransIClassifier()
        else:
            self.i_classifier = IClassifier()
        self.b_classifier = BClassifier()

    def forward(self, x):
        feats, classes = self.i_classifier(x)
        prediction_bag, A, B, A_raw = self.b_classifier(feats, classes)

        return classes, prediction_bag, feats, A, B, A_raw


class Classifier(nn.Module):
    def __init__(self, pooling="mean"):
        super(Classifier, self).__init__()
        self.pooling_mode = pooling
        self.mlp = nn.Linear(100, 1)

    def forward(self, x):
        if self.pooling_mode == "max":
            x = torch.max(x, dim=0)[0]
        elif self.pooling_mode == "mean":
            x = torch.mean(x, dim=0)
        elif self.pooling_mode == "lse":
            x = torch.logsumexp(x, dim=0)
        else:
            raise ValueError("Invalid pooling mode")

        out = self.mlp(x)
        return out, x


class TransABMIL(pl.LightningModule):
    def __init__(
        self,
        criterion=nn.CrossEntropyLoss(),
        num_classes=2,
        prob_transform=0.5,
        max_epochs=250,
        model_type="TransABMIL",
        log_dir="./logs",
        **kwargs,
    ):
        super(TransABMIL, self).__init__()

        self.save_hyperparameters(ignore=["criterion"])
        self.lr = 0.00001
        self.criterion = criterion
        if "pooling" in model_type:
            self.model = Classifier(pooling=model_type.split("_")[0])
        else:
            self.model = MILNet(**kwargs)
        self.num_classes = num_classes
        self.prob_transform = prob_transform
        self.transform = DataAugmentation(0.0, 1, 0.5)
        self.max_epochs = max_epochs
        self.binacc = Accuracy(task="binary")
        self.AUROC = AUROC(task="binary", num_classes=num_classes)
        self.F1 = F1Score(task="binary", num_classes=num_classes, average="macro")
        self.precision_metric = Precision(
            task="binary", num_classes=num_classes, average="macro"
        )
        self.recall = Recall(task="binary", num_classes=num_classes, average="macro")

        self.data = [{"count": 0, "correct": 0} for i in range(self.num_classes)]
        self.log_path = log_dir
        metrics = MetricCollection(
            [
                self.binacc,
                self.F1,
                self.precision_metric,
                self.recall,
            ]
        )
        self.train_metrics = metrics.clone(prefix="train_")
        self.valid_metrics = metrics.clone(prefix="val_")
        self.test_metrics = metrics.clone(prefix="test_")
        self.validation_step_outputs = []
        self.test_step_outputs = []

    def forward(self, x):
        return self.model(x)

    def on_after_batch_transfer(self, batch, batch_idx):
        x, y = batch[0].double(), batch[1]
        if self.trainer.training:
            # => we perform GPU/Batched data augmentation
            x = (self.transform(x)).double()
        return x, y

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.lr,
            weight_decay=1e-4,
        )
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.max_epochs, eta_min=self.lr / 50
        )
        return [optimizer], [lr_scheduler]

    def training_step(self, batch, batch_idx):
        inputs, labels = batch[0].double(), batch[1].double()
        output = self.model(torch.squeeze(inputs).double())
        classes, bag_prediction = output[0], output[1]
        max_prediction, index = torch.max(classes, 0)
        loss_bag = self.criterion(bag_prediction[0], labels)
        loss_max = self.criterion(max_prediction, labels)
        loss = 0.5 * loss_bag + 0.5 * loss_max
        y_prob = torch.sigmoid(bag_prediction)
        acc = self.binacc(y_prob, labels.unsqueeze(1))

        self.log("train_loss", loss, on_step=True, on_epoch=True, logger=True)
        self.log(
            "train_acc",
            acc,
            on_step=True,
            on_epoch=True,
            logger=True,
            prog_bar=True,
        )

        Y_hat = int(y_prob > 0.5)
        Y = int(labels.unsqueeze(1))
        self.data[Y]["count"] += 1
        self.data[Y]["correct"] += Y_hat == Y

        dic = {
            "loss": loss,
            "acc": acc,
        }
        return dic

    def on_train_epoch_end(self):
        for c in range(self.num_classes):
            count = self.data[c]["count"]
            correct = self.data[c]["correct"]
            if count == 0:
                acc = None
            else:
                acc = float(correct) / count
            print("class {}: acc {}, correct {}/{}".format(c, acc, correct, count))

        self.data = [{"count": 0, "correct": 0} for i in range(self.num_classes)]

    def validation_step(self, batch, batch_idx):
        inputs, labels = batch[0].double(), batch[1].double()
        output = self.model(torch.squeeze(inputs).double())
        classes, bag_prediction = output[0], output[1]
        max_prediction, index = torch.max(classes, 0)
        loss_bag = self.criterion(bag_prediction[0], labels)
        loss_max = self.criterion(max_prediction, labels)
        loss = 0.5 * loss_bag + 0.5 * loss_max
        y_prob = torch.sigmoid(bag_prediction)
        acc = self.binacc(y_prob, labels.unsqueeze(1))

        self.log("val_loss", loss, on_step=True, on_epoch=True, logger=True)
        self.log(
            "val_acc",
            acc,
            on_step=True,
            on_epoch=True,
            logger=True,
            prog_bar=True,
        )
        # ---->acc log
        Y_hat = torch.tensor(int(y_prob > 0.5)).to(self.device)

        Y_hat_c = int(y_prob > 0.5)
        Y_c = int(labels.unsqueeze(1))
        self.data[Y_c]["count"] += 1
        self.data[Y_c]["correct"] += Y_hat_c == Y_c

        results = {
            "logits": bag_prediction,
            "Y_prob": y_prob,
            "Y_hat": Y_hat,
            "label": labels,
        }
        self.validation_step_outputs.append(results)
        return results

    def on_validation_epoch_end(self):
        logits = torch.cat([x["logits"] for x in self.validation_step_outputs], dim=0)
        probs = torch.cat([x["Y_prob"] for x in self.validation_step_outputs], dim=0)
        max_probs = torch.stack([x["Y_hat"] for x in self.validation_step_outputs])
        target = torch.stack([x["label"] for x in self.validation_step_outputs], dim=0)

        # ---->
        self.log(
            "val_loss",
            self.criterion(logits, target),
            prog_bar=True,
            on_epoch=True,
            logger=True,
        )
        self.log(
            "auc",
            self.AUROC(probs, target.squeeze()),
            prog_bar=True,
            on_epoch=True,
            logger=True,
        )
        self.log_dict(
            self.valid_metrics(max_probs.squeeze(), target.squeeze()),
            on_epoch=True,
            logger=True,
        )

        # ---->acc log
        for c in range(self.num_classes):
            count = self.data[c]["count"]
            correct = self.data[c]["correct"]
            if count == 0:
                acc = None
            else:
                acc = float(correct) / count
            print("class {}: acc {}, correct {}/{}".format(c, acc, correct, count))
        self.data = [{"count": 0, "correct": 0} for i in range(self.num_classes)]

        self.validation_step_outputs.clear()

    def test_step(self, batch, batch_idx):
        inputs, labels = batch[0].double(), batch[1].double()
        output = self.model(torch.squeeze(inputs).double())
        classes, bag_prediction = output[0], output[1]
        max_prediction, index = torch.max(classes, 0)
        loss_bag = self.criterion(bag_prediction[0], labels)
        loss_max = self.criterion(max_prediction, labels)
        loss = 0.5 * loss_bag + 0.5 * loss_max

        y_prob = torch.sigmoid(bag_prediction)
        acc = self.binacc(y_prob, labels.unsqueeze(1))

        self.log("test_loss", loss, on_step=True, on_epoch=True, logger=True)
        self.log(
            "test_acc",
            acc,
            on_step=True,
            on_epoch=True,
            logger=True,
            prog_bar=True,
        )
        Y_hat = torch.tensor(int(y_prob > 0.5)).to(self.device)
        Y_hat_c = int(y_prob > 0.5)
        Y_c = int(labels.unsqueeze(1))
        self.data[Y_c]["count"] += 1
        self.data[Y_c]["correct"] += Y_hat_c == Y_c

        results = {
            "logits": bag_prediction,
            "Y_prob": y_prob,
            "Y_hat": Y_hat,
            "label": labels,
        }
        self.test_step_outputs.append(results)

        return results

    def on_test_end(self):
        probs = torch.cat([x["Y_prob"] for x in self.test_step_outputs], dim=0)
        max_probs = torch.stack([x["Y_hat"] for x in self.test_step_outputs])
        target = torch.stack([x["label"] for x in self.test_step_outputs], dim=0)

        # ---->
        auc = self.AUROC(probs, target.squeeze())
        metrics = self.test_metrics(max_probs.squeeze(), target.squeeze())
        metrics["auc"] = auc
        for keys, values in metrics.items():
            print(f"{keys} = {values}")
            metrics[keys] = values.cpu().numpy()
        print()
        # ---->acc log
        for c in range(self.num_classes):
            count = self.data[c]["count"]
            correct = self.data[c]["correct"]
            if count == 0:
                acc = None
            else:
                acc = float(correct) / count
            print("class {}: acc {}, correct {}/{}".format(c, acc, correct, count))
        self.data = [{"count": 0, "correct": 0} for i in range(self.num_classes)]
        # ---->
        result = pd.DataFrame([metrics])
        result.to_csv(self.log_path + "/result.csv")

        self.test_step_outputs.clear()
