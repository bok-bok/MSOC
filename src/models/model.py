import numpy as np
import torch
from torch import nn

from models import load_ast, load_av_model, load_vivit


class EnsembleModel(nn.Module):
    def __init__(
        self, decision_type: str, frame_num: int = 30, pretrained: bool = False, device: str = "cuda:1"
    ):
        super(EnsembleModel, self).__init__()
        if decision_type not in ["mv", "asf", "sf", "ff"]:
            """
            mv: majority voting
            asf: average score fusion
            sf: score fusion
            ff: feature fusion
            """
            raise Exception(f"Invalid decision type {decision_type}")

        self.decision_type = decision_type
        self.device = device
        self.softmax = nn.Softmax(dim=1)

        # decision model config
        self.asf_threshold = 0.5

        # general config
        output_dim = 2

        # audio config
        input_fdim = 26
        input_tdim = frame_num * 4

        # video config
        frame_size = 224
        patch_size = 16

        # init model
        self.audio_model = load_ast(output_dim, input_fdim, input_tdim, pretrained=pretrained, device=device)
        self.video_model = load_vivit(
            frame_size, patch_size, output_dim, frame_num, pretrained=pretrained, device=device
        )

        self.audio_visual_model = load_av_model(pretrained=pretrained, device=device)

    def forward(self, frames, lip_frames, spectrogram, stacked_spectrogram):
        # audio
        # input dim : [B, T, F]
        # output dim : [B, 2] feature dim : [B, 768]
        audio_output, audio_feature = self.audio_model(spectrogram)

        # video
        # input dim : [B, T, C, H, W]
        # output dim : [B, 2] feature dim : [B, 192]
        video_output, video_feature = self.video_model(frames)

        # audio visual
        # audio input dim : [B, F, T]
        # video input dim : [B, C, T, H, W]
        # feature dim : [B, 768]
        print(f"Lip frames shape: {lip_frames.shape}")
        print(f"Stacked spectrogram shape: {stacked_spectrogram.shape}")
        audio_visual_output, audio_visual_feature = self.audio_visual_model(lip_frames, stacked_spectrogram)

        print(f"Audio feature shape: {audio_feature.shape}")
        print(f"Audio output shape: {audio_output.shape}")
        print(audio_output)

        print(f"Video feature shape: {video_feature.shape}")
        print(f"Video output shape: {video_output.shape}")
        print(video_output)

        print(f"Audio Visual feature shape: {audio_visual_feature.shape}")
        print(f"Audio Visual output shape: {audio_visual_output.shape}")
        print(audio_visual_output)

        audio_logit = self.softmax(audio_output)
        video_logit = self.softmax(video_output)
        audio_visual_logit = self.softmax(audio_visual_output)

        audio_score = audio_logit.argmax(dim=1)
        video_score = video_logit.argmax(dim=1)
        audio_visual_score = audio_visual_logit.argmax(dim=1)

        if self.decision_type == "mv":
            return self.majority_voting_decision_module(audio_score, video_score, audio_visual_score)
        elif self.decision_type == "asf":
            return self.average_score_fusion_decision_module(audio_score, video_score, audio_visual_score)

    def majority_voting_decision_module(self, audio_score, video_score, audio_visual_score):
        """
        Majority voting decision module
        Args:
        audio_score - torch.Tensor of shape [B]
        video_score - torch.Tensor of shape [B]
        audio_visual_score - torch.Tensor of shape [B]
        Returns:
        score - torch.Tensor of shape [B]
        """

        score = torch.stack([audio_score, video_score, audio_visual_score], dim=1)
        score = torch.mode(score, dim=1)[0]
        return score

    def average_score_fusion_decision_module(self, audio_score, video_score, audio_visual_score):
        """
        Average score fusion decision module
        Args:
        audio_score - torch.Tensor of shape [B]
        video_score - torch.Tensor of shape [B]
        audio_visual_score - torch.Tensor of shape [B]
        Returns:
        score - torch.Tensor of shape [B]
        """
        score = (audio_score + video_score + audio_visual_score) / 3
        score = 1 if score >= self.asf_threshold else 0
        score = (score > self.asf_threshold).float()
        return score

    def create_checkpoint(self, epoch, optimizer, scheduler, loss, path):
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "loss": loss,
        }
        torch.save(checkpoint, path)

    def stacker(feats, stack_order):
        """
        Concatenating consecutive audio frames
        Args:
        feats - numpy.ndarray of shape [T, F]
        stack_order - int (number of neighboring frames to concatenate
        Returns:
        feats - numpy.ndarray of shape [T', F']
        """
        feat_dim = feats.shape[1]
        if len(feats) % stack_order != 0:
            res = stack_order - len(feats) % stack_order
            res = np.zeros([res, feat_dim]).astype(np.float32)
            feats = np.concatenate([feats, res], axis=0)
        feats = feats.reshape((-1, stack_order, feat_dim)).reshape(-1, stack_order * feat_dim)
        return feats
