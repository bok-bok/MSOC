import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import CosineSimilarity, Module, MSELoss
from torch.nn.modules.loss import CrossEntropyLoss


class LossComputer:
    def __init__(self, margin_type, config, device="cuda"):
        self.contrast_loss_fn = ContrastLoss(
            loss_fn=nn.CosineSimilarity(dim=-1), margin=config["margin_contrast"]
        )
        self.margin_type = margin_type

        if margin_type == "margin":
            self.loss_audio = MarginLoss(loss_fn=nn.CosineSimilarity(dim=-1), margin=config["margin_audio"])
            self.loss_video = MarginLoss(loss_fn=nn.CosineSimilarity(dim=-1), margin=config["margin_visual"])
        elif margin_type == "oc":
            self.loss_audio = OCSoftmax(feat_dim=768, alpha=config["alpha"]).to(device)
            self.loss_video = OCSoftmax(feat_dim=768, alpha=config["alpha"]).to(device)

        self.mm_cls = CrossEntropyLoss()

    def compute_loss(
        self, m_logits, v_feats, a_feats, v_embeds, a_embeds, v_label, a_label, c_label, m_label
    ):
        contrast_loss = self.contrast_loss_fn(v_feats, a_feats, c_label)
        v_loss = self.loss_video(v_embeds, v_label)
        a_loss = self.loss_audio(a_embeds, a_label)
        if self.margin_type == "oc":
            v_loss, _ = v_loss
            a_loss, _ = a_loss

        mm_loss = self.mm_cls(m_logits, m_label)
        loss = mm_loss + a_loss + v_loss + contrast_loss

        return {"loss": loss, "mm_loss": mm_loss}


class OCSoftmax(nn.Module):
    def __init__(self, feat_dim=2, r_real=0.9, r_fake=0.2, alpha=20.0):
        super(OCSoftmax, self).__init__()
        self.feat_dim = feat_dim
        self.r_real = r_real
        self.r_fake = r_fake
        self.alpha = alpha
        self.center = nn.Parameter(torch.randn(1, self.feat_dim))
        nn.init.kaiming_uniform_(self.center, 0.25)
        self.softplus = nn.Softplus()

    def forward(self, x, labels):
        """
        Args:
            x: feature matrix with shape (batch_size, feat_dim).
            labels: ground truth labels with shape (batch_size).
        """
        w = F.normalize(self.center, p=2, dim=1)
        x = F.normalize(x, p=2, dim=1)

        scores = x @ w.transpose(0, 1)
        output_scores = scores.clone()

        scores[labels == 0] = self.r_real - scores[labels == 0]
        scores[labels == 1] = scores[labels == 1] - self.r_fake

        loss = self.softplus(self.alpha * scores).mean()

        return loss, output_scores.squeeze(1)


class MarginLoss(Module):
    def __init__(self, loss_fn: Module, margin: float = 0.0):
        super().__init__()
        self.loss_fn = loss_fn
        self.margin = margin

    def forward(self, embeds: Tensor, labels: Tensor):
        loss = []
        for shape_i in range(embeds.shape[0]):
            input_i = embeds[shape_i]
            label_i = labels[shape_i]
            for shape_j in range(embeds.shape[0]):
                input_j = embeds[shape_j]
                label_j = labels[shape_j]
                d = self.loss_fn(input_i, input_j)
                if label_i == label_j:
                    loss.append(1 - d)
                else:
                    loss.append(torch.clip((d - self.margin), min=0.0))
        return torch.mean(torch.stack(loss))


class ContrastLoss(Module):

    def __init__(self, loss_fn: Module, margin: float = 0.0):
        super().__init__()
        self.margin = margin
        self.loss_fn = loss_fn

    def forward(self, pred1: Tensor, pred2: Tensor, labels: Tensor):
        # input: (B, C, T)
        loss = []
        for i in range(pred1.shape[0]):
            # mean L2 distance squared
            d = self.loss_fn(pred1[i, :], pred2[i, :])
            # d = self.cosim(pred1[i, :], pred2[i, :])
            if labels[i]:
                # if is positive pair, minimize distance
                loss.append(1 - d)
            else:
                # if is negative pair, minimize (margin - distance) if distance < margin
                loss.append(torch.clip((d - self.margin), min=0.0))
        return torch.mean(torch.stack(loss))
