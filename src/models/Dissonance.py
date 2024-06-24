from typing import Dict, List, Optional, Sequence, Union

import numpy as np
import torch
import torch.nn as nn
import torchmetrics
from pytorch_lightning import LightningModule

# from lightning import LightningModule
from torch import Tensor
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

import fairseq
import models.avhubert.hubert as hubert
import models.avhubert.hubert_pretraining as hubert_pretraining
import wandb
from fairseq.data.dictionary import Dictionary
from fairseq.modules import LayerNorm
from models.ACM_MM_2020.model import Audio_RNN
from models.ACM_MM_2020.utils import (
    AverageMeter,
    ConfusionMeter,
    calc_accuracy,
    calc_loss,
    denorm,
    get_pred,
    save_checkpoint,
    write_log,
)
from util.eval_metrics import compute_eer
from util.loss import ContrastLoss, MarginLoss


def Average(lst):
    return sum(lst) / len(lst)


def Opposite(a):
    a = a + 1
    a[a > 1.5] = 0
    return a


class Dissonance(LightningModule):

    def __init__(
        self,
        weight_decay=0.0001,
        learning_rate=0.0002,
        distributed=False,
        batch_size=64,
    ):
        super().__init__()

        self.batch_size = batch_size

        self.model = Audio_RNN(img_dim=100, network="resnet18", num_layers_in_fc_layers=512)

        self.weight_decay = weight_decay
        self.learning_rate = learning_rate
        self.distributed = distributed

        self.acc = torchmetrics.classification.BinaryAccuracy()
        self.auroc = torchmetrics.classification.BinaryAUROC(thresholds=None)
        self.f1score = torchmetrics.classification.BinaryF1Score()
        self.recall = torchmetrics.classification.BinaryRecall()
        self.precisions = torchmetrics.classification.BinaryPrecision()

        self.best_loss = 1e9
        self.best_acc, self.best_auroc = 0.0, 0.0
        self.best_real_f1score, self.best_real_recall, self.best_real_precision = 0.0, 0.0, 0.0
        self.best_fake_f1score, self.best_fake_recall, self.best_fake_precision = 0.0, 0.0, 0.0

        self.criterion = nn.CrossEntropyLoss()
        # self.threshold = 0.9
        # self.hyper_param = 0.99
        self.threshold = 1.5
        self.hyper_param = 1.99

        self.softmax = nn.Softmax(dim=1)

    def forward(self, video: Tensor, audio: Tensor):
        audio = audio.transpose(1, 2)
        audio = audio.unsqueeze(1).unsqueeze(1)

        video = video.unsqueeze(1)

        vid_out = self.model.forward_lip(video)
        aud_out = self.model.forward_aud(audio)

        vid_class = self.model.final_classification_lip(vid_out)
        aud_class = self.model.final_classification_aud(aud_out)

        return vid_out, aud_out, vid_class, aud_class
        # return m_logits, v_cross_embeds, a_cross_embeds, v_embeds, a_embeds

    def loss_fn(self, vid_out, aud_out, vid_class, aud_class, m_label) -> Dict[str, Tensor]:

        loss1 = calc_loss(vid_out, aud_out, m_label, self.hyper_param)
        loss2 = self.criterion(vid_class, m_label)
        loss3 = self.criterion(aud_class, m_label)

        loss = loss1 + loss2 + loss3
        return {"loss": loss}

    def training_step(
        self,
        batch: Optional[Union[Tensor, Sequence[Tensor]]] = None,
        batch_idx: Optional[int] = None,
        optimizer_idx: Optional[int] = None,
        hiddens: Optional[Tensor] = None,
    ) -> Tensor:
        vid_out, aud_out, vid_class, aud_class = self(batch["video"], batch["audio"])
        loss_dict = self.loss_fn(
            vid_out,
            aud_out,
            vid_class,
            aud_class,
            batch["m_label"],
        )

        self.log_dict(
            {f"train_{k}": v for k, v in loss_dict.items()},
            on_step=True,
            on_epoch=True,
            prog_bar=False,
            sync_dist=self.distributed,
            batch_size=self.batch_size,
        )

        preds = get_pred(vid_out, aud_out, self.threshold)

        return {
            "loss": loss_dict["loss"],
            "preds": preds.detach().cpu(),
            "targets": batch["m_label"].detach().cpu(),
        }

    def validation_step(
        self,
        batch: Optional[Union[Tensor, Sequence[Tensor]]] = None,
        batch_idx: Optional[int] = None,
        optimizer_idx: Optional[int] = None,
        hiddens: Optional[Tensor] = None,
    ) -> Tensor:

        vid_out, aud_out, vid_class, aud_class = self(batch["video"], batch["audio"])
        loss_dict = self.loss_fn(
            vid_out,
            aud_out,
            vid_class,
            aud_class,
            batch["m_label"],
        )
        preds = get_pred(vid_out, aud_out, self.threshold)

        self.log_dict(
            {f"val_{k}": v for k, v in loss_dict.items()},
            on_step=True,
            on_epoch=True,
            prog_bar=False,
            sync_dist=self.distributed,
            batch_size=self.batch_size,
        )

        return {
            "loss": loss_dict["loss"],
            "preds": preds.detach().cpu(),
            "targets": batch["m_label"].detach().cpu(),
        }

    def test_step(
        self,
        batch: Optional[Union[Tensor, Sequence[Tensor]]] = None,
        batch_idx: Optional[int] = None,
        optimizer_idx: Optional[int] = None,
        hiddens: Optional[Tensor] = None,
    ) -> Tensor:

        vid_out, aud_out, vid_class, aud_class = self(batch["video"], batch["audio"])
        loss_dict = self.loss_fn(
            vid_out,
            aud_out,
            vid_class,
            aud_class,
            batch["m_label"],
        )

        # common and multi-class
        preds = get_pred(vid_out, aud_out, self.threshold)

        return {
            "loss": loss_dict["loss"],
            "preds": preds.detach().cpu(),
            "targets": batch["m_label"].detach().cpu(),
        }

    def training_step_end(self, training_step_outputs):
        # others: common, ensemble, multi-label
        train_acc = self.acc(training_step_outputs["preds"], training_step_outputs["targets"]).item()
        train_auroc = self.auroc(training_step_outputs["preds"], training_step_outputs["targets"]).item()

        self.log("train_acc", train_acc, prog_bar=True, batch_size=self.batch_size)
        self.log("train_auroc", train_auroc, prog_bar=True, batch_size=self.batch_size)

    def validation_step_end(self, validation_step_outputs):
        # others: common, ensemble, multi-label
        val_acc = self.acc(validation_step_outputs["preds"], validation_step_outputs["targets"]).item()
        val_auroc = self.auroc(validation_step_outputs["preds"], validation_step_outputs["targets"]).item()

        self.log("val_re", val_acc + val_auroc, prog_bar=True, batch_size=self.batch_size)
        self.log("val_acc", val_acc, prog_bar=True, batch_size=self.batch_size)
        # self.log("val_auroc", val_auroc, prog_bar=True, batch_size=self.batch_size)

    def training_epoch_end(self, training_step_outputs):
        train_loss = Average([i["loss"] for i in training_step_outputs]).item()
        preds = [item for list in training_step_outputs for item in list["preds"]]
        targets = [item for list in training_step_outputs for item in list["targets"]]
        preds = torch.stack(preds, dim=0)
        targets = torch.stack(targets, dim=0)

        train_acc = self.acc(preds, targets).item()
        train_auroc = self.auroc(preds, targets).item()

        print("Train - loss:", train_loss, "acc: ", train_acc, "auroc: ", train_auroc)

    def validation_epoch_end(self, validation_step_outputs):
        valid_loss = Average([i["loss"] for i in validation_step_outputs]).item()
        preds = [item for list in validation_step_outputs for item in list["preds"]]
        targets = [item for list in validation_step_outputs for item in list["targets"]]
        # scores = [item for list in validation_step_outputs for item in list["scores"]]

        preds = torch.stack(preds, dim=0)
        targets = torch.stack(targets, dim=0)

        # scores = np.stack(scores, axis=0)
        # numpy_targets = targets.cpu().numpy()
        # val_eer = compute_eer(scores[numpy_targets == 1], scores[numpy_targets == 0])[0]

        # if valid_loss <= self.best_loss:
        self.best_acc = self.acc(preds, targets).item()
        self.best_auroc = self.auroc(preds, targets).item()
        self.best_real_f1score = self.f1score(preds, targets).item()
        self.best_real_recall = self.recall(preds, targets).item()
        self.best_real_precision = self.precisions(preds, targets).item()

        self.best_fake_f1score = self.f1score(Opposite(preds), Opposite(targets)).item()
        self.best_fake_recall = self.recall(Opposite(preds), Opposite(targets)).item()
        self.best_fake_precision = self.precisions(Opposite(preds), Opposite(targets)).item()

        self.log("val_real_f1score", self.best_real_f1score, batch_size=self.batch_size)
        self.log("val_fake_f1score", self.best_fake_f1score, batch_size=self.batch_size)
        self.log("val_auroc", self.best_auroc, batch_size=self.batch_size)
        # self.log("val_eer", val_eer, batch_size=self.batch_size)

        self.best_loss = valid_loss
        print(
            "Valid loss: ",
            self.best_loss,
            "acc: ",
            self.best_acc,
            "auroc: ",
            self.best_auroc,
            "real_f1score:",
            self.best_real_f1score,
            "real_recall: ",
            self.best_real_recall,
            "real_precision: ",
            self.best_real_precision,
            "fake_f1score: ",
            self.best_fake_f1score,
            "fake_recall: ",
            self.best_fake_recall,
            "fake_precision: ",
            self.best_fake_precision,
        )

    def test_epoch_end(self, test_step_outputs):
        test_loss = Average([i["loss"] for i in test_step_outputs]).item()
        preds = [item for list in test_step_outputs for item in list["preds"]]
        targets = [item for list in test_step_outputs for item in list["targets"]]
        # scores = [item for list in test_step_outputs for item in list["scores"]]

        preds = torch.stack(preds, dim=0)
        targets = torch.stack(targets, dim=0)
        # scores = torch.stack(scores, dim=0)
        # scores is numpy
        # scores = np.stack(scores, axis=0)
        # numpy_targets = targets.cpu().numpy()
        # test_eer = compute_eer(scores[numpy_targets == 1], scores[numpy_targets == 0])[0]

        test_acc = self.acc(preds, targets).item()
        test_auroc = self.auroc(preds, targets).item()
        test_real_f1score = self.f1score(preds, targets).item()
        test_real_recall = self.recall(preds, targets).item()
        test_real_precision = self.precisions(preds, targets).item()

        test_fake_f1score = self.f1score(Opposite(preds), Opposite(targets)).item()
        test_fake_recall = self.recall(Opposite(preds), Opposite(targets)).item()
        test_fake_precision = self.precisions(Opposite(preds), Opposite(targets)).item()

        # self.log("test_eer", test_eer, batch_size=self.batch_size)
        self.log("test_acc", test_acc, batch_size=self.batch_size)
        self.log("test_auroc", test_auroc, batch_size=self.batch_size)
        self.log("test_real_f1score", test_real_f1score, batch_size=self.batch_size)
        self.log("test_real_recall", test_real_recall, batch_size=self.batch_size)
        self.log("test_real_precision", test_real_precision, batch_size=self.batch_size)
        self.log("test_fake_f1score", test_fake_f1score, batch_size=self.batch_size)
        self.log("test_fake_recall", test_fake_recall, batch_size=self.batch_size)
        self.log("test_fake_precision", test_fake_precision, batch_size=self.batch_size)
        return {
            "loss": test_loss,
            "test_acc": test_acc,
            "auroc": test_auroc,
            "real_f1score": test_real_f1score,
            "real_recall": test_real_recall,
            "real_precision": test_real_precision,
            "fake_f1score": test_fake_f1score,
            "fake_recall": test_fake_recall,
            "fake_precision": test_fake_precision,
        }

    def configure_optimizers(self):

        optimizer = Adam(
            self.parameters(), lr=self.learning_rate, betas=(0.5, 0.9), weight_decay=self.weight_decay
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": ReduceLROnPlateau(optimizer, factor=0.5, patience=3, verbose=True, min_lr=1e-8),
                "monitor": "val_loss",
            },
        }
