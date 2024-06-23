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

# from src.models.archive.attentionLayer import attentionLayer
from models.talkNet.audioEncoder import audioEncoder
from models.talkNet.resnet import ResEncoder
from models.talkNet.visualEncoder import visualConv1D, visualFrontend, visualTCN
from util.eval_metrics import compute_eer
from util.loss import OCSoftmax


def Average(lst):
    return sum(lst) / len(lst)


def Opposite(a):
    a = a + 1
    a[a > 1.5] = 0
    return a


class MSOC(LightningModule):
    def __init__(
        self,
        positional_emb_flag=True,
        weight_decay=0.0001,
        learning_rate=0.0002,
        batch_size=64,
        audio_threshold=0.5,
        visual_threshold=0.5,
        final_threshold=0.5,
        pred_strategy="mean",
        share_oc=False,
        scnet=False,
        middle_infer=False,
        save_score=False,
        test_subset="C",
        seed=42,
    ):
        super(MSOC, self).__init__()
        self.scnet = scnet
        self.seed = seed
        if middle_infer:
            print("Using middle infer")
        self.acc = torchmetrics.classification.BinaryAccuracy()
        self.auroc = torchmetrics.classification.BinaryAUROC(thresholds=None)
        self.f1score = torchmetrics.classification.BinaryF1Score()
        self.recall = torchmetrics.classification.BinaryRecall()
        self.precisions = torchmetrics.classification.BinaryPrecision()

        self.best_loss = 1e9
        self.best_acc, self.best_auroc = 0.0, 0.0
        self.best_real_f1score, self.best_real_recall, self.best_real_precision = 0.0, 0.0, 0.0
        self.best_fake_f1score, self.best_fake_recall, self.best_fake_precision = 0.0, 0.0, 0.0

        self.test_subset = test_subset
        self.save_score = save_score
        self.scores = {
            "a_scores": [],
            "v_scores": [],
            "av_scores": [],
            # "final_scores": [],
            "a_labels": [],
            "v_labels": [],
            "av_labels": [],
        }

        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.batch_size = batch_size

        self.audio_threshold = audio_threshold
        self.visual_threshold = visual_threshold
        self.final_threshold = final_threshold
        self.pred_strategy = pred_strategy

        self.share_oc = share_oc
        if self.share_oc:
            print("Using share oc")
        self.middle_infer = middle_infer
        # AV
        # Visual Temporal Encoder
        # self.av_visualFrontend = visualFrontend()  # Visual Frontend
        if self.scnet:
            self.av_visualFrontend = scnet50_v1d(30)
            self.v_visualFrontend = scnet50_v1d(30)
        else:
            self.av_visualFrontend = ResEncoder(relu_type="relu", light=True)
            # self.av_visualTCN = visualTCN()  # Visual Temporal Network TCN
            self.av_visualConv1D = visualConv1D()  # Visual Temporal Network Conv1d
            # Visual branch
            self.v_visualFrontend = ResEncoder(relu_type="relu", light=True)
            # self.v_visualTCN = visualTCN()  # Visual Temporal Network TCN
            self.v_visualConv1D = visualConv1D()  # Visual Temporal Network Conv1d

        # Audio Temporal Encoder
        self.av_audioEncoder = audioEncoder(layers=[2, 3, 4, 2], num_filters=[16, 32, 64, 128])

        # Audio branch
        self.a_audioEncoder = audioEncoder(layers=[2, 3, 4, 2], num_filters=[16, 32, 64, 128])

        # Visual branch
        # self.v_visualFrontend = visualFrontend(light=True)  # Visual Frontend

        audio_dim = 128
        if self.scnet:
            visual_dim = 512
        else:
            visual_dim = 128
        embed_dim = audio_dim + visual_dim
        # loss
        self.loss_audio = OCSoftmax(feat_dim=audio_dim).to(self.device)
        self.loss_video = OCSoftmax(feat_dim=visual_dim).to(self.device)

        self.av_loss_audio = OCSoftmax(feat_dim=audio_dim).to(self.device)
        self.av_loss_video = OCSoftmax(feat_dim=visual_dim).to(self.device)

        self.mm_cls = CrossEntropyLoss()
        self.binary_cls = BCEWithLogitsLoss()
        # self.binary_cls = nn.BCELoss()
        self.softmax = nn.Softmax(dim=1)

        # Audio-visual Backend
        self.av_classifier = nn.Sequential(
            nn.Linear(embed_dim, embed_dim), nn.ReLU(inplace=True), nn.Linear(embed_dim, 2)
        )

    def forward(self, video, audio):
        # Single branch
        # Extract features from the audio and video
        v_features = self.forward_visual_frontend(video)
        a_features = self.forward_audio_frontend(audio)

        # a_features = self.forward_audio_backend(audio_feat)
        # v_features = self.forward_visual_backend(video_feat)

        a_embed = a_features.mean(dim=1)
        v_embed = v_features.mean(dim=1)

        # Audio-Visual
        av_audio_feat = self.forward_av_audio_frontend(audio)
        av_video_feat = self.forward_av_visual_frondend(video)

        av_audio_embed = av_audio_feat.mean(dim=1)
        av_video_embed = av_video_feat.mean(dim=1)

        av_features = self.forward_audio_visual_backend(av_audio_feat, av_video_feat)
        av_features = av_features.mean(dim=1)

        av_logits = self.av_classifier(av_features)

        return av_logits, av_video_embed, av_audio_embed, v_embed, a_embed

    def forward_visual_frontend(self, x):
        if self.scnet:
            B, C, T, H, W = x.shape
            x = x.view(B, C * T, H, W)
            x = x.view((-1, 3) + x.size()[2:])
            # x = self.model(x)
            x = self.v_visualFrontend(x)
            x = x.view(B, T, -1)

        else:
            x = self.v_visualFrontend(x)
            # x = x.view(B, T, 512)
            x = x.transpose(1, 2)
            x = self.v_visualConv1D(x)
            x = x.transpose(1, 2)
        return x

    def forward_audio_frontend(self, x):
        x = x.unsqueeze(1).transpose(2, 3)
        x = self.a_audioEncoder(x)
        return x

    def forward_av_visual_frondend(self, x):
        if self.scnet:
            B, C, T, H, W = x.shape
            x = x.view(B, C * T, H, W)
            x = x.view((-1, 3) + x.size()[2:])
            # x = self.model(x)
            x = self.av_visualFrontend(x)
            x = x.view(B, T, -1)

        else:
            x = self.av_visualFrontend(x)
            # x = x.view(B, T, 512)
            x = x.transpose(1, 2)
            x = self.av_visualConv1D(x)
            x = x.transpose(1, 2)
        return x

    def forward_av_audio_frontend(self, x):
        x = x.unsqueeze(1).transpose(2, 3)
        x = self.av_audioEncoder(x)
        return x

    def forward_audio_visual_backend(self, x1, x2):
        x = torch.cat((x1, x2), 2)
        return x

    def forward_audio_backend(self, x):
        return x

    def forward_visual_backend(self, x):
        return x

    def loss_fn(self, av_logits, av_v_feats, av_a_feats, v_feats, a_feats, v_label, a_label, m_label):
        v_loss, v_score = self.loss_video(v_feats, v_label)
        a_loss, a_score = self.loss_audio(a_feats, a_label)

        av_v_loss, av_v_score = self.av_loss_video(av_v_feats, v_label)
        av_a_loss, av_a_score = self.av_loss_audio(av_a_feats, a_label)

        av_softmax = self.softmax(av_logits)

        # av_value = av_logits[:, 1]
        # value for train
        # change the value of the score to 0-1
        train_v_value = (v_score + 1) / 2
        train_a_value = (a_score + 1) / 2
        # train_final_value = ((train_v_value + train_a_value + av_value) / 3).float()
        # mm_loss = self.binary_cls(train_final_value, m_label.float())

        mm_loss = self.mm_cls(av_logits, m_label)

        av_value = av_softmax[:, 1]
        if self.save_score:
            self.scores["a_scores"].extend(a_score.data.cpu().numpy())
            self.scores["v_scores"].extend(v_score.data.cpu().numpy())
            self.scores["av_scores"].extend(av_value.data.cpu().numpy())
            self.scores["a_labels"].extend(a_label.data.cpu().numpy())
            self.scores["v_labels"].extend(v_label.data.cpu().numpy())
            self.scores["av_labels"].extend(m_label.data.cpu().numpy())

        # use av score for inference if middle infer
        if self.middle_infer:
            v_value = (av_v_score > self.visual_threshold).float()
            a_value = (av_a_score > self.audio_threshold).float()
        else:
            v_value = (v_score > self.visual_threshold).float()
            a_value = (a_score > self.audio_threshold).float()

        final_value = ((v_value + a_value + av_value) / 3).float()
        # if self.middle_infer:
        #     preds = torch.argmax(self.softmax(av_logits), dim=1)
        # else:

        if self.pred_strategy == "mean":
            preds = (final_value > self.final_threshold).float()
        elif self.pred_strategy == "min":

            av_preds = (av_value > self.final_threshold).float()
            min_va = torch.min(v_value, a_value)
            preds = torch.min(min_va, av_preds)

        total_a_loss = a_loss + av_a_loss
        total_v_loss = v_loss + av_v_loss
        # av_logits = self.av_classifier(av_feature)
        # mm_loss = self.mm_cls(av_logits, m_label)
        scores = final_value.data.cpu().numpy()
        # loss = total_v_loss + total_a_loss + mm_loss + av_mm_loss
        loss = total_v_loss + total_a_loss + mm_loss
        loss_dict = {
            "loss": loss,
            "mm_loss": mm_loss,
            "a_loss": a_loss,
            "v_loss": v_loss,
            "av_a_loss": av_a_loss,
            "av_v_loss": av_v_loss,
            # "av_mm_loss": av_mm_loss,
            # "v_mm_loss": v_mm_loss,
            # "a_mm_loss": a_mm_loss,
        }
        result_dict = {"loss_dict": loss_dict, "preds": preds, "scores": scores}
        return result_dict

    def training_step(
        self,
        batch: Optional[Union[Tensor, Sequence[Tensor]]] = None,
        batch_idx: Optional[int] = None,
        optimizer_idx: Optional[int] = None,
        hiddens: Optional[Tensor] = None,
    ) -> Tensor:
        av_logits, av_v_feats, av_a_feats, v_feats, a_feats = self(batch["video"], batch["audio"])
        result_dict = self.loss_fn(
            av_logits,
            av_v_feats,
            av_a_feats,
            v_feats,
            a_feats,
            batch["v_label"],
            batch["a_label"],
            batch["m_label"],
        )
        loss_dict = result_dict["loss_dict"]

        # common and multi-class
        # preds = torch.argmax(self.softmax(av_logits), dim=1)
        preds = result_dict["preds"]
        #
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
        av_logits, av_v_feats, av_a_feats, v_feats, a_feats = self(batch["video"], batch["audio"])

        result_dict = self.loss_fn(
            av_logits,
            av_v_feats,
            av_a_feats,
            v_feats,
            a_feats,
            batch["v_label"],
            batch["a_label"],
            batch["m_label"],
        )

        loss_dict = result_dict["loss_dict"]

        preds = result_dict["preds"]
        scores = result_dict["scores"]

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
        av_logits, av_v_feats, av_a_feats, v_feats, a_feats = self(batch["video"], batch["audio"])

        result_dict = self.loss_fn(
            av_logits,
            av_v_feats,
            av_a_feats,
            v_feats,
            a_feats,
            batch["v_label"],
            batch["a_label"],
            batch["m_label"],
        )

        # common and multi-class
        # scores = self.softmax(av_logits)
        # preds = torch.argmax(scores, dim=1)
        # scores = scores[:, 1].data.cpu().numpy()
        loss_dict = result_dict["loss_dict"]

        preds = result_dict["preds"]
        scores = result_dict["scores"]

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

        if self.save_score:
            name = "talknetSF"
            if self.scnet:
                name += "_scnet"

            name += f"_{self.seed}_{self.test_subset}_scores"
            path = f"Scores/{name}.npy"
            print(f"Saving scores to {path}")

            np.save(path, self.scores, allow_pickle=True)

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
