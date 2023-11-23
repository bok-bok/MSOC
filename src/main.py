import sys

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

import eval_metrics as em
from audio import OCSoftmax, ResNet
from utils import SwanDataset

if __name__ == "__main__":
    train_dataset = SwanDataset("train")
    test_dataset = SwanDataset("test3")
    device = "cuda:1"
    enc_dim = 256
    epochs = 10

    lr = 0.0003

    train_dataloader = DataLoader(
        train_dataset, batch_size=64, shuffle=False, num_workers=8, pin_memory=True
    )
    test_dataloader = DataLoader(
        test_dataset, batch_size=64, shuffle=True, num_workers=8, pin_memory=True
    )
    # data shape
    # [64, 1, 40, 750]

    criterion = nn.CrossEntropyLoss()

    # set ocsoftmax function
    ocsoftmax = OCSoftmax(feat_dim=256).to(device)
    ocsoftmax.train()
    ocsoftmax_optimzer = torch.optim.SGD(ocsoftmax.parameters(), lr=lr)

    lfcc_model = ResNet(3, enc_dim, resnet_type="18", nclasses=2).to(device)
    lfcc_optimizer = torch.optim.Adam(lfcc_model.parameters(), lr=lr, weight_decay=0.0005)

    for epoch in range(epochs):
        lfcc_model.train()
        total_loss = 0
        for data, labels in tqdm(train_dataloader):
            data = data.to(device)
            labels = labels.to(device)

            feats, outputs = lfcc_model(data)
            lfcc_loss = criterion(outputs, labels)

            # oc softmax
            ocsoftmaxloss, _ = ocsoftmax(feats, labels)
            lfcc_loss = ocsoftmaxloss
            lfcc_optimizer.zero_grad()
            ocsoftmax_optimzer.zero_grad()
            lfcc_loss.backward()
            lfcc_optimizer.step()
            ocsoftmax_optimzer.step()
            total_loss += lfcc_loss.item()

        lfcc_model.eval()
        with torch.no_grad():
            idx_loader, score_loader = [], []

            for data, labels in tqdm(test_dataloader):
                data = data.to(device)
                labels = labels.to(device)

                feats, outputs = lfcc_model(data)

                ocsoftmaxloss, score = ocsoftmax(feats, labels)
                score_loader.append(score)
                idx_loader.append(labels)

        scores = torch.cat(score_loader, 0).data.cpu().numpy()
        labels = torch.cat(idx_loader, 0).data.cpu().numpy()
        val_eer = em.compute_eer(scores[labels == 0], scores[labels == 1])[0]

        print(f"epoch: {epoch}, train_loss : {total_loss}, val_eer: {val_eer}")
