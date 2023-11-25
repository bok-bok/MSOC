import sys

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm

import eval_metrics as em
from audio import OCSoftmax, ResNet
from utils import SwanDataset, plot_pca

if __name__ == "__main__":
    early_stop_cnt = 0
    out_folder = "audio/weights"
    train_dataset = SwanDataset("train")
    test_dataset = SwanDataset("test3")
    device = "cuda:0"
    enc_dim = 256
    epochs = 100
    devices = [0, 3]
    lr = 0.00001
    prev_eer = 1e8

    train_dataloader = DataLoader(
        train_dataset, batch_size=128, shuffle=False, num_workers=8, pin_memory=True
    )
    test_dataloader = DataLoader(
        test_dataset, batch_size=128, shuffle=True, num_workers=8, pin_memory=True
    )
    # data shape
    # [64, 1, 40, 750]

    criterion = nn.CrossEntropyLoss()

    # set ocsoftmax function
    ocsoftmax = OCSoftmax(feat_dim=256).to(device)
    ocsoftmax.train()
    ocsoftmax_optimzer = torch.optim.SGD(ocsoftmax.parameters(), lr=lr)
    ocsoftmax_scheduler = CosineAnnealingLR(ocsoftmax_optimzer, T_max=epochs, eta_min=0.00001)

    lfcc_model = ResNet(3, enc_dim, resnet_type="18", nclasses=2).to(device)
    lfcc_model = nn.DataParallel(lfcc_model, device_ids=devices)

    lfcc_optimizer = torch.optim.AdamW(lfcc_model.parameters(), lr=lr, weight_decay=0.0005)
    lfcc_scheduler = CosineAnnealingLR(lfcc_optimizer, T_max=epochs, eta_min=0.00001)

    for epoch in range(epochs):
        lfcc_model.train()
        total_loss = 0
        train_correct = 0
        train_total = 0
        for data, labels in tqdm(train_dataloader):
            data = data.to(device)
            labels = labels.to(device)

            feats, outputs = lfcc_model(data)
            train_correct += (outputs.argmax(1) == labels).sum().item()
            train_total += labels.size(0)

            # lfcc_loss = criterion(outputs, labels)

            # oc softmax
            ocsoftmaxloss, _ = ocsoftmax(feats, labels)
            lfcc_loss = ocsoftmaxloss
            ocsoftmax_optimzer.zero_grad()
            lfcc_optimizer.zero_grad()
            lfcc_loss.backward()
            lfcc_optimizer.step()
            ocsoftmax_optimzer.step()
            total_loss += lfcc_loss.item()

        lfcc_model.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            idx_loader, score_loader, features_loader = [], [], []

            for data, labels in test_dataloader:
                data = data.to(device)
                labels = labels.to(device)

                feats, outputs = lfcc_model(data)
                correct += (outputs.argmax(1) == labels).sum().item()
                total += labels.size(0)

                ocsoftmaxloss, score = ocsoftmax(feats, labels)

                features_loader.append(feats)
                score_loader.append(score)
                idx_loader.append(labels)

        scores = torch.cat(score_loader, 0).data.cpu().numpy()
        labels = torch.cat(idx_loader, 0).data.cpu().numpy()
        features = torch.cat(features_loader, dim=0).data.cpu().numpy()

        plot_pca(features=features, labels=labels, epoch=epoch)

        val_eer = em.compute_eer(scores[labels == 0], scores[labels == 1])[0]

        print(
            f"epoch: {epoch}, train_loss : {total_loss} , val_eer: {val_eer}, cnt : {early_stop_cnt}"
        )
        if val_eer < prev_eer:
            prev_eer = val_eer
            torch.save(lfcc_model.state_dict(), f"{out_folder}/lfcc_model.pt")
            torch.save(ocsoftmax.state_dict(), f"{out_folder}/ocsoftmax.pt")
            early_stop_cnt = 0
        else:
            early_stop_cnt += 1

        if early_stop_cnt == 10:
            break

        lfcc_scheduler.step()
        ocsoftmax_scheduler.step()
