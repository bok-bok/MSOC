import os
from typing import Dict, List, Optional, Sequence, Union

import numpy as np
import torch
import torch.nn as nn
import torchmetrics
from pytorch_lightning import LightningModule
from torch import Tensor
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

from models.SCNet import scnet50_v1d
from models.talkNet.audioEncoder import audioEncoder
from models.talkNet.resnet import ResEncoder
from models.talkNet.visualEncoder import visualConv1D
from util.eval_metrics import compute_eer
from util.loss import OCSoftmax


def Average(lst):
    return sum(lst) / len(lst)


def Opposite(a):
    a = a + 1
    a[a > 1.5] = 0
    return a


class AVOC(LightningModule):
    def __init__(
        self,
        positional_emb_flag=True,
        weight_decay=0.0001,
        learning_rate=0.0002,
        batch_size=64,
        oc_option="both",
        scnet=False,
        save_features=False,
        score_fusion=False,
    ):
        super(AVOC, self).__init__()
        # self.light = True
        self.score_fusion = score_fusion
        if oc_option not in ["no", "audio", "video", "both"]:
            raise ValueError("Invalid OC option")
        self.oc_option = oc_option
        self.save_features = save_features
        self.features = {
            "a_features": [],
            "v_features": [],
            "a_labels": [],
            "v_labels": [],
            "av_labels": [],
        }
        print(f"OC option: {self.oc_option}")

        self.scnet = scnet

        self.acc = torchmetrics.classification.BinaryAccuracy()
        self.auroc = torchmetrics.classification.BinaryAUROC(thresholds=None)
        self.f1score = torchmetrics.classification.BinaryF1Score()
        self.recall = torchmetrics.classification.BinaryRecall()
        self.precisions = torchmetrics.classification.BinaryPrecision()

        self.best_loss = 1e9
        self.best_acc, self.best_auroc = 0.0, 0.0
        self.best_real_f1score, self.best_real_recall, self.best_real_precision = 0.0, 0.0, 0.0
        self.best_fake_f1score, self.best_fake_recall, self.best_fake_precision = 0.0, 0.0, 0.0

        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.batch_size = batch_size

        if scnet:
            self.visualFrontend = scnet50_v1d(30)
        else:
            self.visualFrontend = ResEncoder(relu_type="relu", light=True)

            self.visualConv1D = visualConv1D()  # Visual Temporal Network Conv1d

        # Audio Temporal Encoder
        self.audioEncoder = audioEncoder(layers=[2, 3, 4, 2], num_filters=[16, 32, 64, 128])

        audio_dim = 128
        if self.scnet:
            # visual_dim = 1024
            visual_dim = 512
        else:
            visual_dim = 128

        embed_dim = audio_dim + visual_dim
        self.av_classifier = nn.Sequential(
            nn.Linear(embed_dim, embed_dim), nn.ReLU(inplace=True), nn.Linear(embed_dim, 2)
        )

        # loss
        self.loss_audio = OCSoftmax(feat_dim=audio_dim).to(self.device)
        self.loss_video = OCSoftmax(feat_dim=visual_dim).to(self.device)
        self.mm_cls = CrossEntropyLoss()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, video, audio):

        # Extract features from the audio and video
        v_features = self.forward_visual_frontend(video)
        a_features = self.forward_audio_frontend(audio)

        a_features = a_features.mean(dim=1)
        # if not self.resnet3d:
        v_features = v_features.mean(dim=1)
        av_features = torch.cat((a_features, v_features), 1)
        # av_features = av_features.mean(dim=1)
        av_logits = self.av_classifier(av_features)

        # print(av_features.shape, a_features.shape, v_features.shape)

        return av_logits, v_features, a_features

    def forward_visual_frontend(self, x):
        # B, T, W, H = x.shape
        # x = x.view(B * T, 1, 1, W, H)
        # x = (x / 255 - 0.4161) / 0.1688
        if self.scnet:
            B, C, T, H, W = x.shape
            x = x.view(B, C * T, H, W)
            x = x.view((-1, 3) + x.size()[2:])
            # x = self.model(x)
            x = self.visualFrontend(x)
            x = x.view(B, T, -1)

        else:
            x = self.visualFrontend(x)
            # x = x.view(B, T, 512)
            x = x.transpose(1, 2)
            x = self.visualConv1D(x)
            x = x.transpose(1, 2)
        return x

    def forward_audio_frontend(self, x):
        # x = x.transpose(1, 2)
        # x = unstack_features(x)

        x = x.unsqueeze(1).transpose(2, 3)
        x = self.audioEncoder(x)
        return x

    def forward_cross_attention(self, x1, x2):
        x1_c = self.crossA2V(src=x1, tar=x2)
        x2_c = self.crossV2A(src=x2, tar=x1)
        return x1_c, x2_c

    def forward_audio_visual_backend(self, x1, x2):
        x = torch.cat((x1, x2), 2)
        if self.transformer:
            x = self.selfAV(src=x)
        else:
            x = self.selfAV(src=x, tar=x)
        # x = torch.reshape(x, (-1, 256))
        return x

    def forward_audio_backend(self, x):
        if self.projection:
            x = self.project_audio(x)
        return x

    def forward_visual_backend(self, x):
        if self.projection:
            x = self.project_video(x)
        return x

    def loss_fn(self, av_logits, v_feats, a_feats, v_label, a_label, m_label):
        v_loss, v_score = self.loss_video(v_feats, v_label)
        a_loss, a_score = self.loss_audio(a_feats, a_label)

        # av_logits = self.av_classifier(av_feature)
        mm_loss = self.mm_cls(av_logits, m_label)
        if self.oc_option == "no":
            loss = mm_loss
        elif self.oc_option == "audio":
            loss = a_loss + mm_loss
        elif self.oc_option == "video":
            loss = v_loss + mm_loss
        elif self.oc_option == "both":
            loss = v_loss + a_loss + mm_loss

        if self.save_features:
            self.features["a_features"].extend(a_feats.data.cpu().numpy())
            self.features["v_features"].extend(v_feats.data.cpu().numpy())
            self.features["a_labels"].extend(a_label.data.cpu().numpy())
            self.features["v_labels"].extend(v_label.data.cpu().numpy())
            self.features["av_labels"].extend(m_label.data.cpu().numpy())
        av_softmax = self.softmax(av_logits)
        if self.score_fusion:
            av_value = av_softmax[:, 1]
            final_value = ((av_value + v_score + a_score) / 3).float()
            preds = (final_value > 0.5).float()

        else:
            preds = torch.argmax(av_softmax, dim=1)
        loss_dict = {"loss": loss, "mm_loss": mm_loss, "v_loss": v_loss, "a_loss": a_loss}
        result_dict = {"loss_dict": loss_dict, "preds": preds}
        return result_dict

    def training_step(
        self,
        batch: Optional[Union[Tensor, Sequence[Tensor]]] = None,
        batch_idx: Optional[int] = None,
        optimizer_idx: Optional[int] = None,
        hiddens: Optional[Tensor] = None,
    ) -> Tensor:
        av_logits, v_feats, a_feats = self(batch["video"], batch["audio"])
        result_dict = self.loss_fn(
            av_logits,
            v_feats,
            a_feats,
            batch["v_label"],
            batch["a_label"],
            batch["m_label"],
        )

        # common and multi-class
        # preds = torch.argmax(self.softmax(av_logits), dim=1)
        preds = result_dict["preds"]
        #
        loss_dict = result_dict["loss_dict"]
        self.log_dict(
            {f"train_{k}": v for k, v in loss_dict.items()},
            on_step=True,
            on_epoch=True,
            prog_bar=False,
            sync_dist=False,
            batch_size=self.batch_size,
        )

        return {"loss": loss_dict["loss"], "preds": preds.detach(), "targets": batch["m_label"].detach()}

    def validation_step(
        self,
        batch: Optional[Union[Tensor, Sequence[Tensor]]] = None,
        batch_idx: Optional[int] = None,
        optimizer_idx: Optional[int] = None,
        hiddens: Optional[Tensor] = None,
    ) -> Tensor:
        av_logits, v_feats, a_feats = self(batch["video"], batch["audio"])
        result_dict = self.loss_fn(
            av_logits,
            v_feats,
            a_feats,
            batch["v_label"],
            batch["a_label"],
            batch["m_label"],
        )

        # common and multi-class
        scores = self.softmax(av_logits)
        # preds = torch.argmax(scores, dim=1)
        scores = scores[:, 1].data.cpu().numpy()
        preds = result_dict["preds"]

        loss_dict = result_dict["loss_dict"]
        self.log_dict(
            {f"val_{k}": v for k, v in loss_dict.items()},
            on_step=True,
            on_epoch=True,
            prog_bar=False,
            sync_dist=False,
            batch_size=self.batch_size,
        )

        return {
            "loss": loss_dict["mm_loss"],
            "preds": preds.detach(),
            "targets": batch["m_label"].detach(),
            "scores": scores,
        }

    def test_step(
        self,
        batch: Optional[Union[Tensor, Sequence[Tensor]]] = None,
        batch_idx: Optional[int] = None,
        optimizer_idx: Optional[int] = None,
        hiddens: Optional[Tensor] = None,
    ) -> Tensor:
        av_logits, v_feats, a_feats = self(batch["video"], batch["audio"])
        result_dict = self.loss_fn(
            av_logits,
            v_feats,
            a_feats,
            batch["v_label"],
            batch["a_label"],
            batch["m_label"],
        )

        # common and multi-class
        scores = self.softmax(av_logits)
        # preds = torch.argmax(scores, dim=1)
        scores = scores[:, 1].data.cpu().numpy()
        preds = result_dict["preds"]

        loss_dict = result_dict["loss_dict"]
        return {
            "loss": loss_dict["mm_loss"],
            "preds": preds.detach(),
            "targets": batch["m_label"].detach(),
            "scores": scores,
        }

    def training_step_end(self, training_step_outputs):
        # others: common, ensemble, multi-label
        train_acc = self.acc(training_step_outputs["preds"], training_step_outputs["targets"]).item()
        train_auroc = self.auroc(training_step_outputs["preds"], training_step_outputs["targets"]).item()

        self.log("train_acc", train_acc, prog_bar=True, batch_size=self.batch_size)
        self.log("train_auroc", train_auroc, prog_bar=True, batch_size=self.batch_size)

    def validation_step_end(self, validation_step_outputs):
        # others: common, ensemble, multi-label
        scores = validation_step_outputs["scores"]
        scores = torch.tensor(scores)

        val_acc = self.acc(validation_step_outputs["preds"], validation_step_outputs["targets"]).item()
        # val_auroc = self.auroc(scores, validation_step_outputs["targets"].cpu()).item()
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
        scores = [item for list in validation_step_outputs for item in list["scores"]]

        preds = torch.stack(preds, dim=0)
        targets = torch.stack(targets, dim=0)

        scores = np.stack(scores, axis=0)
        numpy_targets = targets.cpu().numpy()
        val_eer = compute_eer(scores[numpy_targets == 1], scores[numpy_targets == 0])[0]

        # if valid_loss <= self.best_loss:
        self.best_acc = self.acc(preds, targets).item()
        self.best_auroc = self.auroc(preds, targets).item()
        self.best_real_f1score = self.f1score(preds, targets).item()
        self.best_real_recall = self.recall(preds, targets).item()
        self.best_real_precision = self.precisions(preds, targets).item()

        self.best_fake_f1score = self.f1score(Opposite(preds), Opposite(targets)).item()
        self.best_fake_recall = self.recall(Opposite(preds), Opposite(targets)).item()
        self.best_fake_precision = self.precisions(Opposite(preds), Opposite(targets)).item()

        self.log("val_eer", val_eer, batch_size=self.batch_size)
        self.log("val_real_f1score", self.best_real_f1score, batch_size=self.batch_size)
        self.log("val_fake_f1score", self.best_fake_f1score, batch_size=self.batch_size)
        self.log("val_auroc", self.best_auroc, batch_size=self.batch_size)

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
        scores = [item for list in test_step_outputs for item in list["scores"]]

        preds = torch.stack(preds, dim=0)
        targets = torch.stack(targets, dim=0)

        scores = np.stack(scores, axis=0)
        numpy_targets = targets.cpu().numpy()
        test_eer = compute_eer(scores[numpy_targets == 1], scores[numpy_targets == 0])[0]
        scores = torch.tensor(scores)

        test_acc = self.acc(preds, targets).item()
        # test_auroc = self.auroc(preds, targets).item()
        test_auroc = self.auroc(scores, targets.cpu()).item()
        test_real_f1score = self.f1score(preds, targets).item()
        test_real_recall = self.recall(preds, targets).item()
        test_real_precision = self.precisions(preds, targets).item()

        test_fake_f1score = self.f1score(Opposite(preds), Opposite(targets)).item()
        test_fake_recall = self.recall(Opposite(preds), Opposite(targets)).item()
        test_fake_precision = self.precisions(Opposite(preds), Opposite(targets)).item()

        if self.save_features:
            name = "talknet"
            if self.oc_option == "no":
                name += "_no"
            else:
                name += "_both"
            path = "/data/kyungbok/outputs/features"
            if not os.path.exists(path):
                os.makedirs(path)
            print(f"Saving features length of {len(self.features['a_labels'])}")
            np.save(f"{path}/{name}.npy", self.features, allow_pickle=True)

        self.log("test_eer", test_eer, batch_size=self.batch_size)
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
