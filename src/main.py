import argparse
import logging
import os
import sys

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm

import eval_metrics as em
import wandb
from datasets import FakeAVDataset, SwanDataset
from engine_lr import get_outputs_feats
from loss import OCSoftmax
from models import ASTModel, ViViT

# from models.model import EnsembleModel
# from util import plot_pca


def get_args():
    parser = argparse.ArgumentParser()

    # training
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.0003)
    parser.add_argument("--lr_decay", type=float, default=0.5, help="lr decay rate")
    parser.add_argument("--interval", type=int, default=10, help="interval to decay lr")

    parser.add_argument("--beta_1", type=float, default=0.9, help="bata_1 for Adam")
    parser.add_argument("--beta_2", type=float, default=0.999, help="beta_2 for Adam")
    parser.add_argument("--eps", type=float, default=1e-8, help="epsilon for Adam")

    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs for training")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument(
        "--add_loss",
        type=str,
        default="ocsoftmax",
        choices=["softmax", "ocsoftmax"],
        help="loss for one-class training",
    )
    parser.add_argument("--r_real", type=float, default=0.9)
    parser.add_argument("--r_fake", type=float, default=0.2)
    parser.add_argument("--alpha", type=float, default=20, help="scale factor for ocsoftmax")
    # parser.add_argument("--enc_dim", type=int, default=768, help="encoder dimension")

    parser.add_argument("--gpu", type=str, help="GPU index", default="2")
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)

    # Dataset
    parser.add_argument("--dataset", type=str, default="fakeavceleb")
    parser.add_argument("--frame_num", type=int, default=30)

    # Model
    parser.add_argument("--classifier_type", type=str, default="A", choices=["A", "V", "AV"])
    # parser.add_argument("--decision_type", type=str, default="mv", choices=["mv", "asf", "sf", "ff"])

    args = parser.parse_args()

    # set device
    args.device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    return args


def adjust_learning_rate(args, optimizer, epoch_num):
    lr = args.lr * (args.lr_decay ** (epoch_num // args.interval))
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


if __name__ == "__main__":
    args = get_args()
    prev_loss = None
    WAB = True
    early_stop_cnt = 0

    # get dataset
    if args.dataset == "fakeavceleb":
        train_dataset = FakeAVDataset("train", frame_num=args.frame_num, classifier_type=args.classifier_type)
        test_dataset = FakeAVDataset("test", frame_num=args.frame_num, classifier_type=args.classifier_type)
    elif args.dataset == "swan":
        train_dataset = SwanDataset("train")
        test_dataset = SwanDataset("test3")

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    test_dataloader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True
    )
    save_dir = "weights"
    # set model
    if args.classifier_type == "A":
        model = ASTModel(label_dim=2, input_fdim=26, input_tdim=args.frame_num * 4).to(args.device)
        args.enc_dim = 768
        save_dir += "/ast"
    elif args.classifier_type == "V":
        model = ViViT(image_size=224, patch_size=16, num_classes=2, num_frames=args.frame_num).to(args.device)
        args.enc_dim = 192
        save_dir += "/vivit"
    else:
        # model = AVModel(pretrained=False, device=args.device)
        pass
    if WAB:
        run = wandb.init(
            project=args.classifier_type,
            config={
                "add_loss": args.add_loss,
                "lr": args.lr,
                "batch_size": args.batch_size,
                "frame_num": args.frame_num,
                "classifier_type": args.classifier_type,
            },
        )

    criterion = nn.CrossEntropyLoss()

    # set softmax
    if args.add_loss == "ocsoftmax":
        print("init ocsoftmax")
        ocsoftmax = OCSoftmax(args.enc_dim, r_real=args.r_real, r_fake=args.r_fake, alpha=args.alpha).to(
            args.device
        )
        ocsoftmax.train()
        ocsoftmax_optimzer = torch.optim.SGD(ocsoftmax.parameters(), lr=args.lr)
        save_dir += "/ocsoftmax"
    elif args.add_loss == "softmax":
        print("init softmax")
        criterion = nn.CrossEntropyLoss()
        save_dir += "/softmax"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, betas=(args.beta_1, args.beta_2), eps=args.eps, weight_decay=0.0005
    )

    for epoch in range(args.epochs):
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        # train_idx_loader, train_features_loader, train_scores = [], [], []
        adjust_learning_rate(args, optimizer, epoch)
        if args.add_loss == "ocsoftmax":
            adjust_learning_rate(args, ocsoftmax_optimzer, epoch)

        for idx, (data) in tqdm(enumerate(train_dataloader)):
            # get output
            labels = data[-1]
            labels = labels.to(args.device)
            outputs, feats = get_outputs_feats(model, data, args)

            # compute loss and backprop
            if args.add_loss == "ocsoftmax":
                loss, scores = ocsoftmax(feats, labels)
                outputs = (scores > 0).float()
                train_correct += (outputs == labels).sum().item()

                optimizer.zero_grad()
                ocsoftmax_optimzer.zero_grad()

                loss.backward()
                ocsoftmax_optimzer.step()
                optimizer.step()

            elif args.add_loss == "softmax":
                train_correct += (outputs.argmax(1) == labels).sum().item()
                optimizer.zero_grad()
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
            train_total += labels.size(0)

            train_loss += loss.item()

        train_acc = round(train_correct * 100 / train_total, 2)

        model.eval()
        with torch.no_grad():
            val_correct = 0
            val_total = 0
            val_loss = 0
            for idx, (data) in tqdm(enumerate(test_dataloader)):
                labels = data[-1].to(args.device)
                outputs, feats = get_outputs_feats(model, data, args)
                if args.add_loss == "ocsoftmax":
                    ocsoftmaxloss, score = ocsoftmax(feats, labels)
                    val_loss += ocsoftmaxloss.item()
                    correct = (score > 0).float().eq(labels).sum().item()

                elif args.add_loss == "softmax":
                    correct = (outputs.argmax(1) == labels).sum().item()
                val_correct += correct
                val_total += labels.size(0)

        val_acc = round(val_correct * 100 / val_total, 2)
        if WAB:
            wandb.log(
                {
                    "train_loss": train_loss,
                    "train_acc": train_acc,
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                    "correct": val_correct,
                }
            )
        print(f"Epoch {epoch}: train loss {train_loss}, train acc {train_acc}, val acc {val_acc}")
        if prev_loss is None:
            prev_loss = val_loss
            torch.save(model, f"{save_dir}/model.pth")
            if args.add_loss == "ocsoftmax":
                torch.save(ocsoftmax, f"{save_dir}/ocsoftmax.pth")

        elif val_loss < prev_loss:
            torch.save(model, f"{save_dir}/model.pth")
            if args.add_loss == "ocsoftmax":
                torch.save(ocsoftmax, f"{save_dir}/ocsoftmax.pth")

            prev_loss = val_loss
            early_stop_cnt = 0
        else:
            early_stop_cnt += 1

        if early_stop_cnt == 20:
            break
