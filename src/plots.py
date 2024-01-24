import argparse
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets import FakeAVDataset, SwanDataset
from loss import OCSoftmax
from models import ASTModel, ViViT


def remove_module_from_state_dict(state_dict):
    new_state_dict = {}
    for key, value in state_dict.items():
        if "module" in key:
            new_state_dict[key.replace("module.", "")] = value
        else:
            new_state_dict[key] = value
    return new_state_dict


def visualize_dev_and_eval(dev_feat, dev_labels, eval_feat, eval_labels, center, seed, out_fold):
    if not os.path.exists(out_fold):
        os.makedirs(out_fold)

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(8, 8))
    # plt.ion()
    # c = ['#ff0000', '#ffff00', '#00ff00', '#00ffff', '#0000ff',
    #      '#ff00ff', '#990000', '#999900', '#009900', '#009999']
    c = ["#ff0000", "#003366", "#ffff00"]

    # plt.clf()
    torch.manual_seed(668)
    num_centers, enc_dim = center.shape
    dev_num = 3000
    eval_num = 3000
    ind_dev = torch.randperm(dev_feat.shape[0])[:dev_num].numpy()
    ind_eval = torch.randperm(eval_feat.shape[0])[:eval_num].numpy()

    print(f"dev_feat : {dev_feat.shape}")
    print(f"eval_feat : {eval_feat.shape}")
    dev_feat_sample = dev_feat[ind_dev]
    eval_feat_sample = eval_feat[ind_eval]
    dev_lab_sam = dev_labels[ind_dev]
    eval_lab_sam = eval_labels[ind_eval]

    if enc_dim > 2:
        print(f"center : {center.shape}")
        print(f"dev_feat_sample : {dev_feat_sample.shape}")
        print(f"eval_feat_sample : {eval_feat_sample.shape}")
        X = np.concatenate((center, dev_feat_sample, eval_feat_sample), axis=0)
        os.environ["PYTHONHASHSEED"] = str(668)
        np.random.seed(668)
        X_tsne = TSNE(random_state=seed, perplexity=40, early_exaggeration=40).fit_transform(X)
        center = X_tsne[:num_centers]
        feat_dev = X_tsne[num_centers : num_centers + dev_num]
        feat_eval = X_tsne[num_centers + eval_num :]
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X)
        ex_ratio = pca.explained_variance_ratio_
        center_pca = X_pca[:num_centers]
        feat_pca_dev = X_pca[num_centers : num_centers + dev_num]
        feat_pca_eval = X_pca[num_centers + eval_num :]
    else:
        center_pca = center
        feat_dev = dev_feat_sample
        feat_eval = eval_feat_sample
        feat_pca_dev = feat_dev
        feat_pca_eval = feat_eval
        ex_ratio = [0.5, 0.5]
    # t-SNE visualization

    ax1.plot(feat_dev[dev_lab_sam == 0, 0], feat_dev[dev_lab_sam == 0, 1], ".", c=c[0], markersize=1.2)
    ax1.plot(feat_dev[dev_lab_sam == 1, 0], feat_dev[dev_lab_sam == 1, 1], ".", c=c[1], markersize=1.2)

    #     ax1.plot(feat_dev[dev_tag_sam == 17, 0], feat_dev[dev_tag_sam == 17, 1], '.', c='olive', markersize=1)
    ax1.axis("off")
    # ax1.plot(center[:, 0], center[:, 1], 'x', c=c[2], markersize=5)
    # ax2.plot(center[:, 0], center[:, 1], 'x', c=c[2], markersize=5)
    plt.setp((ax2), xlim=ax1.get_xlim(), ylim=ax1.get_ylim())
    ax2.plot(
        feat_eval[eval_lab_sam == 0, 0],
        feat_eval[eval_lab_sam == 0, 1],
        ".",
        c=c[0],
        markersize=1.2,
    )
    ax2.plot(
        feat_eval[eval_lab_sam == 1, 0],
        feat_eval[eval_lab_sam == 1, 1],
        ".",
        c=c[1],
        markersize=1.2,
    )
    # ax2.plot(feat_eval[eval_tag_sam == 17, 0], feat_dev[eval_tag_sam == 17, 1], '.', c='darkgreen', markersize=1.5)
    ax2.axis("off")
    # ax1.legend(['genuine', 'spoofing', 'center'])
    # PCA visualization
    ax3.plot(
        feat_pca_dev[dev_lab_sam == 0, 0],
        feat_pca_dev[dev_lab_sam == 0, 1],
        ".",
        c=c[0],
        markersize=1.2,
    )
    ax3.plot(
        feat_pca_dev[dev_lab_sam == 1, 0],
        feat_pca_dev[dev_lab_sam == 1, 1],
        ".",
        c=c[1],
        markersize=1.2,
    )
    # ax3.plot(center_pca[:, 0], center_pca[:, 1], 'x', c=c[2], markersize=5)
    ax3.axis("off")
    plt.setp((ax4), xlim=ax3.get_xlim(), ylim=ax3.get_ylim())
    ax4.plot(
        feat_pca_eval[eval_lab_sam == 0, 0],
        feat_pca_eval[eval_lab_sam == 0, 1],
        ".",
        c=c[0],
        markersize=1.2,
    )
    ax4.plot(
        feat_pca_eval[eval_lab_sam == 1, 0],
        feat_pca_eval[eval_lab_sam == 1, 1],
        ".",
        c=c[1],
        markersize=1.2,
    )
    # ax4.plot(feat_pca_eval[eval_tag_sam == 17, 0], feat_pca_dev[eval_tag_sam == 17, 1], '.', c='darkgreen', markersize=1.5)
    ax4.axis("off")
    # ax4.legend(['genuine', 'spoofing', 'center'])
    plt.savefig(os.path.join(out_fold, "_vis_feat.jpg"))
    fig.clf()

    # new figure
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(8, 8))

    ax1.plot(feat_dev[dev_lab_sam == 0, 0], feat_dev[dev_lab_sam == 0, 1], ".", c=c[0], markersize=1.2)
    #     ax1.plot(feat_dev[dev_tag_sam == 17, 0], feat_dev[dev_tag_sam == 17, 1], '.', c='olive', markersize=1)
    ax1.axis("off")
    # ax1.plot(center[:, 0], center[:, 1], 'x', c=c[2], markersize=5)
    # ax2.plot(center[:, 0], center[:, 1], 'x', c=c[2], markersize=5)
    plt.setp((ax2), xlim=ax1.get_xlim(), ylim=ax1.get_ylim())
    ax2.plot(
        feat_eval[eval_lab_sam == 0, 0],
        feat_eval[eval_lab_sam == 0, 1],
        ".",
        c=c[0],
        markersize=1.2,
    )

    # ax2.plot(feat_eval[eval_tag_sam == 17, 0], feat_dev[eval_tag_sam == 17, 1], '.', c='darkgreen', markersize=1.5)
    ax2.axis("off")
    # ax1.legend(['genuine', 'spoofing', 'center'])
    # PCA visualization
    ax3.plot(
        feat_pca_dev[dev_lab_sam == 0, 0],
        feat_pca_dev[dev_lab_sam == 0, 1],
        ".",
        c=c[0],
        markersize=1.2,
    )

    # ax3.plot(center_pca[:, 0], center_pca[:, 1], 'x', c=c[2], markersize=5)
    ax3.axis("off")
    plt.setp((ax4), xlim=ax3.get_xlim(), ylim=ax3.get_ylim())
    ax4.plot(
        feat_pca_eval[eval_lab_sam == 0, 0],
        feat_pca_eval[eval_lab_sam == 0, 1],
        ".",
        c=c[0],
        markersize=1.2,
    )

    # ax4.plot(feat_pca_eval[eval_tag_sam == 17, 0], feat_pca_dev[eval_tag_sam == 17, 1], '.', c='darkgreen', markersize=1.5)
    ax4.axis("off")
    plt.savefig(os.path.join(out_fold, "_vis_feat_original.jpg"))

    plt.close(fig)


def get_features(model, dataLoader, device):
    model.eval()
    # feature_loader, label_loader = [], []
    features, labels = None, None
    with torch.no_grad():
        for data, label in tqdm(dataLoader):
            data = data.to(device)
            _, feats = model(data)
            feats = feats.view(feats.size(0), -1)

            # feature_loader.append(feats.detach().cpu().numpy())
            # label_loader.append((labels.detach().cpu().numpy()))
            feats = feats.detach().cpu()
            label = label.detach().cpu()

            features = torch.cat((features, feats), 0) if features is not None else feats
            labels = torch.cat((labels, label), 0) if labels is not None else label
    print(f"features shape: {features.shape}")
    print(f"labels shape: {labels.shape}")

    # print("start concat")
    # features = np.concatenate(feature_loader, 0)
    # labels = np.concatenate(label_loader, 0)
    # print("end concat")

    return features, labels


def get_args():
    parser = argparse.ArgumentParser()

    # training
    parser.add_argument("--frame_num", type=int, default=30)
    parser.add_argument("--classifier_type", type=str, default="A", choices=["A", "V", "AV"])

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()

    model_path = f"weights"
    if args.classifier_type == "A":
        model_path += "/ast"
    else:
        model_path += "/vivit"

    print(f"loading model from {model_path}")
    model = torch.load(f"{model_path}/ocsoftmax/model.pth")
    ocsoftmax = torch.load(f"{model_path}/ocsoftmax/ocsoftmax.pth")
    # oc_state_dict = remove_module_from_state_dict(torch.load(loss_model_path))
    # oc_state_dict = torch.load(loss_model_path)
    # ocsoftmax = OCSoftmax(feat_dim=2048 * 30)
    # ocsoftmax.load_state_dict(oc_state_dict)

    center = ocsoftmax.center.detach().cpu().numpy()

    device = "cuda:2"

    # model = Model(num_classes=2).to(device)
    # state_dict = torch.load(feat_model_path)
    # state_dict = remove_module_from_state_dict(state_dict)
    # model.load_state_dict(state_dict)

    train_dataset = FakeAVDataset("train", frame_num=args.frame_num, classifier_type=args.classifier_type)
    test_dataset = FakeAVDataset("test", frame_num=args.frame_num, classifier_type=args.classifier_type)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=128,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
    )
    test_dataloader = DataLoader(test_dataset, batch_size=128, shuffle=True, num_workers=8, pin_memory=True)
    dev_feat, dev_labels = get_features(model, train_dataloader, device=device)
    eval_feat, eval_labels = get_features(model, test_dataloader, device=device)
    visualize_dev_and_eval(dev_feat, dev_labels, eval_feat, eval_labels, center, 78, "data")
