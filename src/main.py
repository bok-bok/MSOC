import sys

import pandas as pd
from torch.utils.data import DataLoader

from model import ResNet
from utils import SwanDataset

if __name__ == "__main__":
    dataset = SwanDataset("train")
    device = "cuda"
    enc_dim = 256

    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
    data , target = next(iter(dataloader))
    # data shape
    # [64, 1, 40, 750]

    print(data.shape)
    print(target.shape)

    lfcc_model = ResNet(3, enc_dim, resnet_type='18', nclasses=2).to(device)
    data = data.to(device)

    out = lfcc_model(data)
    print(out.shape)