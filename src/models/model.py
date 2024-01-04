import torch
from torch import nn

from models import ASTModel, ViViT


class AVModel(nn.Module):
    def __init__(self, frame_num=30):
        super(AVModel, self).__init__()
        # general config
        output_dim = 2

        # audio config
        input_fdim = 80
        input_tdim = frame_num * 4

        # video config
        frame_size = 224
        patch_size = 16

        # init model
        self.audio_model = ASTModel(
            label_dim=output_dim, input_fdim=input_fdim, input_tdim=input_tdim
        )
        self.video_model = ViViT(
            image_size=frame_size,
            patch_size=patch_size,
            num_classes=output_dim,
            num_frames=frame_num,
        )

    def forward(self, frames, spectrogram):
        # audio
        # output dim : [batch, 2] feature dim : [batch, 768]
        audio_output, audio_feature = self.audio_model(spectrogram)

        # video
        # output dim : [batch, 2] feature dim : [batch, 192]
        video_output, video_feature = self.video_model(frames)

        # concat
        concated_feature = torch.cat((audio_feature, video_feature), dim=1)

        output = self.decision_module(audio_output, video_output, concated_feature)
        return output

    def decision_module(self, audio_output, video_output, concated_feature):
        # do something here
        pass
