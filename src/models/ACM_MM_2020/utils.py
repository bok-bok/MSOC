## code source - https://github.com/TengdaHan/DPC/blob/master/utils/utils.py
import glob
import os
import pickle
import re
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import torch

plt.switch_backend("agg")
from collections import deque

from torchvision import transforms
from tqdm import tqdm


def save_checkpoint(state, is_best=0, gap=1, filename="models/checkpoint.pth.tar", keep_all=False):
    torch.save(state, filename)
    last_epoch_path = os.path.join(os.path.dirname(filename), "epoch%s.pth.tar" % str(state["epoch"] - gap))
    if not keep_all:
        try:
            os.remove(last_epoch_path)
        except:
            pass
    if is_best:
        past_best = glob.glob(os.path.join(os.path.dirname(filename), "model_best_*.pth.tar"))
        for i in past_best:
            try:
                os.remove(i)
            except:
                pass
        torch.save(
            state, os.path.join(os.path.dirname(filename), "model_best_epoch%s.pth.tar" % str(state["epoch"]))
        )


def write_log(content, epoch, filename):
    if not os.path.exists(filename):
        log_file = open(filename, "w")
    else:
        log_file = open(filename, "a")
    log_file.write("## Epoch %d:\n" % epoch)
    log_file.write("time: %s\n" % str(datetime.now()))
    log_file.write(content + "\n\n")
    log_file.close()


def calc_accuracy(vid_out, aud_out, target, threshold):
    batch_size = target.size(0)
    pred = 0
    correct = 0
    for batch in range(batch_size):
        dist = torch.dist(vid_out[batch, :].view(-1), aud_out[batch, :].view(-1), 2)
        tar = target[batch, :].view(-1).item()
        if dist < threshold:
            pred = 1
        else:
            pred = 0
        if pred == tar:
            correct += 1
    return correct * (1 / batch_size)


def get_pred(vid_out, aud_out, threshold):
    batch_size = vid_out.size(0)
    pred = []
    dists = []
    real_count = 0
    fake_count = 0
    for batch in range(batch_size):
        dist = torch.dist(vid_out[batch, :].view(-1), aud_out[batch, :].view(-1), 2)
        if dist < threshold:
            pred.append(1)
            real_count += 1
        else:
            pred.append(0)
            fake_count += 1
        dists.append(dist.detach().item())
    pred = torch.tensor(pred)
    min_dist = min(dists)
    # print("Average distance: %0.4f" % np.mean(dists), real_count, fake_count, min_dist)
    return pred


def calc_loss(vid_out, aud_out, target, hyper_param):
    batch_size = target.size(0)
    loss = 0
    # print(vid_out.size(), aud_out.size(), target.size())
    for batch in range(batch_size):
        dist = torch.dist(vid_out[batch, :].view(-1), aud_out[batch, :].view(-1), 2)
        # tar = target[batch, :].view(-1)
        tar = target[batch]
        # real : 1, fake : 0
        loss += (tar * (dist**2)) + ((1 - tar) * (max(hyper_param - dist, 0) ** 2))

    return loss.mul_(1 / batch_size)


def denorm(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    assert len(mean) == len(std) == 3
    inv_mean = [-mean[i] / std[i] for i in range(3)]
    inv_std = [1 / i for i in std]
    return transforms.Normalize(mean=inv_mean, std=inv_std)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.local_history = deque([])
        self.local_avg = 0
        self.history = []
        self.dict = {}  # save all data values here
        self.save_dict = {}  # save mean and std here, for summary table

    def update(self, val, n=1, history=0, step=5):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        if history:
            self.history.append(val)
        if step > 0:
            self.local_history.append(val)
            if len(self.local_history) > step:
                self.local_history.popleft()
            self.local_avg = np.average(self.local_history)

    def dict_update(self, val, key):
        if key in self.dict.keys():
            self.dict[key].append(val)
        else:
            self.dict[key] = [val]

    def __len__(self):
        return self.count


class AccuracyTable(object):
    """compute accuracy for each class"""

    def __init__(self):
        self.dict = {}

    def update(self, pred, tar):
        pred = torch.squeeze(pred)
        tar = torch.squeeze(tar)
        for i, j in zip(pred, tar):
            i = int(i)
            j = int(j)
            if j not in self.dict.keys():
                self.dict[j] = {"count": 0, "correct": 0}
            self.dict[j]["count"] += 1
            if i == j:
                self.dict[j]["correct"] += 1

    def print_table(self, label):
        for key in self.dict.keys():
            acc = self.dict[key]["correct"] / self.dict[key]["count"]
            print(
                "%s: %2d, accuracy: %3d/%3d = %0.6f"
                % (label, key, self.dict[key]["correct"], self.dict[key]["count"], acc)
            )


class ConfusionMeter(object):
    """compute and show confusion matrix"""

    def __init__(self, num_class):
        self.num_class = num_class
        self.mat = np.zeros((num_class, num_class))
        self.precision = []
        self.recall = []

    def update(self, pred, tar):
        self.mat[pred][tar] += 1

    def print_mat(self):
        print("Confusion Matrix: (target in columns)")
        print(self.mat)

    def plot_mat(self, path, dictionary=None, annotate=False):
        plt.figure(dpi=600)
        plt.imshow(
            self.mat,
            cmap=plt.cm.jet,
            interpolation=None,
            extent=(0.5, np.shape(self.mat)[0] + 0.5, np.shape(self.mat)[1] + 0.5, 0.5),
        )
        width, height = self.mat.shape
        if annotate:
            for x in range(width):
                for y in range(height):
                    plt.annotate(
                        str(int(self.mat[x][y])),
                        xy=(y + 1, x + 1),
                        horizontalalignment="center",
                        verticalalignment="center",
                        fontsize=8,
                    )

        if dictionary is not None:
            plt.xticks(
                [i + 1 for i in range(width)], [dictionary[i] for i in range(width)], rotation="vertical"
            )
            plt.yticks([i + 1 for i in range(height)], [dictionary[i] for i in range(height)])
        plt.xlabel("Ground Truth")
        plt.ylabel("Prediction")
        plt.colorbar()
        plt.tight_layout()
        plt.savefig(path, format="svg")
        plt.clf()

        # for i in range(width):
        #     if np.sum(self.mat[i,:]) != 0:
        #         self.precision.append(self.mat[i,i] / np.sum(self.mat[i,:]))
        #     if np.sum(self.mat[:,i]) != 0:
        #         self.recall.append(self.mat[i,i] / np.sum(self.mat[:,i]))
        # print('Average Precision: %0.4f' % np.mean(self.precision))
        # print('Average Recall: %0.4f' % np.mean(self.recall))
