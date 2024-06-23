import numpy as np
import torch
from model import *

video_seq = torch.randn(32, 3, 30, 100, 100)
video_seq = video_seq.unsqueeze(1)

# video_seq = video_seq.view(1, 30, C, H, W).transpose(1,2)
# video_seq = video_seq.transpose(1, 2)
print(video_seq.shape)


audio_seq = torch.randn(32, 120, 13)
audio_seq = audio_seq.transpose(1, 2)
audio_seq = audio_seq.unsqueeze(1).unsqueeze(1)
print(audio_seq.shape)
model = Audio_RNN(img_dim=100, network="resnet18", num_layers_in_fc_layers=1024)
vid_out = model.forward_lip(video_seq)
# aud_out = model.forward_aud(audio_seq)
print(vid_out.shape)


vid_class = model.final_classification_lip(vid_out)
print(vid_class.shape)
# aud_class = model.final_classification_aud(aud_out)
