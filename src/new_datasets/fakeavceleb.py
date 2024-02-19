import os
import random
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from python_speech_features import logfbank
from pytorch_lightning import LightningDataModule
from pytorch_lightning.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
from scipy.io import wavfile
from torch import Tensor
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, RandomSampler
from torch.utils.data.sampler import WeightedRandomSampler

import models.avhubert.utils as custom_utils
from util.constants import FAKE, ORIGINAL


class Fakeavceleb(Dataset):
    def __init__(self, subset, frame_num):
        self.subset = subset  # train, val, test

        self.df = self._get_subset_df(subset)
        self.image_size = 128
        self.image_crop_size = 540
        self.image_mean = 0.421
        self.image_std = 0.165
        self.stack_order_audio = 4
        self.pad_audio = False
        self.scale_percent = 0.5
        self.video_target_frames = frame_num
        self.audio_target_frames = frame_num * 4

        if self.subset in "train":
            self.transform = custom_utils.Compose(
                [
                    custom_utils.Normalize(0.0, 255.0),
                    custom_utils.RandomCrop((100, 100)),
                    custom_utils.HorizontalFlip(0.5),
                    custom_utils.Normalize(self.image_mean, self.image_std),
                ]
            )
        else:
            self.transform = custom_utils.Compose(
                [
                    custom_utils.Normalize(0.0, 255.0),
                    custom_utils.CenterCrop((100, 100)),
                    custom_utils.Normalize(self.image_mean, self.image_std),
                ]
            )

    def __getitem__(self, index):
        meta = self.df.iloc[index]
        # label = meta["label"]

        c_label = ORIGINAL if ("RealVideo" in meta.type) and ("RealAudio" in meta.type) else FAKE
        v_label = ORIGINAL if "RealVideo" in meta.type else FAKE
        a_label = ORIGINAL if "RealAudio" in meta.type else FAKE
        m_label = ORIGINAL if "real" in meta.method else FAKE

        c_label = torch.tensor(c_label)
        v_label = torch.tensor(v_label)
        a_label = torch.tensor(a_label)
        m_label = torch.tensor(m_label)

        video_feats, audio_feats = self.load_feature(meta["preprocessed_directory"])

        with torch.no_grad():
            audio_feats = F.layer_norm(audio_feats, audio_feats.shape[1:])

        return video_feats, audio_feats, c_label, v_label, a_label, m_label

    def load_feature(self, directory):
        video_feats = self._load_frames(directory)
        audio_feats = self._load_audio(directory)

        return video_feats, audio_feats

    def _load_frames(self, directory: str):
        def pil_loader(path):
            with open(path, "rb") as f:
                img = Image.open(f)
                return img.convert("RGB")

        frame_dir = os.path.join(directory, "frames")

        # frame_0.jpg -> 0
        sorted_frames_files = sorted(os.listdir(frame_dir), key=lambda x: int(x.split(".")[0].split("_")[1]))
        frame_num = len(sorted_frames_files)

        # clip frames to target number
        if frame_num > self.video_target_frames:
            sorted_frames_files = sorted_frames_files[: self.video_target_frames]

        # load and transform frames
        frames = [pil_loader(os.path.join(frame_dir, img)) for img in sorted_frames_files]
        frames = np.stack(frames)
        feats = self.transform(frames)

        feats = torch.tensor(feats, dtype=torch.float32)

        # pad frames to target number
        if frame_num < self.video_target_frames:
            p = self.video_target_frames - frame_num
            feats = torch.nn.functional.pad(feats, (0, 0, 0, 0, 0, 0, 0, p))

        feats = feats.permute(3, 0, 1, 2)  # [T, H, W, C] -> [C, T, H, W]
        return feats

    def _load_audio(self, directory: str):
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
                res = np.zeros([res, feat_dim]).astype(feats.dtype)
                feats = np.concatenate([feats, res], axis=0)
            feats = feats.reshape((-1, stack_order, feat_dim)).reshape(-1, stack_order * feat_dim)
            return feats

        audio_dir = os.path.join(directory, "audio.wav")
        sample_rate, wav_data = wavfile.read(audio_dir)
        print(wav_data.shape)
        audio_feats = logfbank(wav_data, samplerate=sample_rate)  # [T, F]
        # change to torch
        audio_feats = torch.from_numpy(audio_feats)
        audio_feats = audio_feats.transpose(1, 0)  # [T, F] -> [F, T]
        # clip audio to target number
        n_frames = audio_feats.shape[1]
        p = self.audio_target_frames - n_frames
        if p > 0:
            # m = torch.nn.ZeroPad2d((0, 0, 0, p))
            m = torch.nn.ZeroPad2d((0, p, 0, 0))
            audio_feats = m(audio_feats)
        elif p < 0:
            audio_feats = audio_feats[:, 0 : self.audio_target_frames]

        audio_feats = audio_feats.transpose(1, 0)  # [F, T] -> [T, F]
        audio_feats = np.array(audio_feats)
        audio_feats = stacker(
            audio_feats, self.stack_order_audio
        )  # [T/stack_order_audio, F*stack_order_audio]

        audio_feats = torch.from_numpy(audio_feats).float()
        audio_feats = audio_feats.transpose(1, 0)  # [T, F] -> [F, T]
        return audio_feats

    def __len__(self):
        return len(self.df)

    def _get_subset_df(self, dataset_type: str) -> pd.DataFrame:
        # Return a subset of the dataframe
        """
        Args:
            subset (str): train, val, test
            classifier_type (str): A, V, AV
        """

        if dataset_type not in ["train", "val", "test"]:
            raise ValueError("dataset_type should be one of train, test")

        def change_dir(df):
            def change_dir_helper(row):
                # change /neil/storage ... to /data/kyungbok
                # write code here
                cur_dir = row["preprocessed_directory"]
                new_dir = cur_dir.replace("storage/neil", "/data/kyungbok")
                return new_dir

            df["preprocessed_directory"] = df.apply(change_dir_helper, axis=1)
            return df

        df = pd.read_csv("new_datasets/data.csv")

        df = change_dir(df)

        method_column_mapping = {
            "wav2lip": "source",
            "rtvc": "source",
            "faceswap": "target1",
            "fsgan": "target1",
            "faceswap-wav2lip": "target1",
            "fsgan-wav2lip": "target1",
            "real": "source",
        }

        df["subject"] = df.apply(lambda row: row[method_column_mapping[row["method"]]], axis=1)
        # set 70 sources for test
        test_source = df["source"].unique()[:70]
        val_source = df["source"].unique()[70:140]
        train_source = df["source"].unique()[140:]

        if "train" in dataset_type:
            df = df[df["subject"].isin(train_source)]
            # df = df[df["source"]].isin(train_source)
            count = 120
        elif "val" in dataset_type:
            df = df[df["subject"].isin(val_source)]
            # df = df[df["source"]].isin(val_source)
            count = 23
        elif "test" in dataset_type:
            df = df[df["subject"].isin(test_source)]
            # df = df[df["source"]].isin(test_source)

            count = 23

        A = df[df["category"] == "A"]
        B = df[df["category"] == "B"]
        C = df[df["category"] == "C"].groupby("method").sample(n=count, random_state=42)
        D = df[df["category"] == "D"].groupby("method").sample(n=count, random_state=42)
        print(f"A: {len(A)}, B: {len(B)}, C: {len(C)}, D: {len(D)}")
        df = pd.concat([A, B, C, D])

        # if "test" in dataset_type:
        #     if dataset_type == "test":
        #         df = df[df["subject"].isin(test_source)]
        #     else:
        #         df = self.create_test_dataset(dataset_type)
        # elif dataset_type == "train":
        #     df = df[df["subject"].isin(train_source)]
        # elif dataset_type == "val":
        #     df = df[df["subject"].isin(val_source)]
        # fakecount = len(df[df["label"] == FAKE])
        # realcount = len(df[df["label"] == ORIGINAL])

        return df


def collater(samples):

    audio, video = [s["audio"] for s in samples], [s["video"] for s in samples]

    if audio[0] is None:
        audio = None
    if video[0] is None:
        video = None
    if audio is not None:
        audio_sizes = [len(s) for s in audio]
    else:
        audio_sizes = [len(s) for s in video]
    # if self.pad_audio:
    #     audio_size = min(max(audio_sizes), self.max_sample_size)
    # else:
    audio_size = min(min(audio_sizes), 500)
    if audio is not None:
        collated_audios, padding_mask, audio_starts = collater_audio(audio, audio_size)
    else:
        collated_audios, audio_starts = None, None
    if video is not None:
        collated_videos, padding_mask, audio_starts = collater_audio(video, audio_size, audio_starts)
    else:
        collated_videos = None


def collater_audio(audios, audio_size, audio_starts=None):
    audio_feat_shape = list(audios[0].shape[1:])
    collated_audios = audios[0].new_zeros([len(audios), audio_size] + audio_feat_shape)
    padding_mask = torch.BoolTensor(len(audios), audio_size).fill_(False)  #
    start_known = audio_starts is not None
    audio_starts = [0 for _ in audios] if not start_known else audio_starts
    for i, audio in enumerate(audios):
        diff = len(audio) - audio_size
        if diff == 0:
            collated_audios[i] = audio
        elif diff < 0:
            assert self.pad_audio
            collated_audios[i] = torch.cat([audio, audio.new_full([-diff] + audio_feat_shape, 0.0)])
            padding_mask[i, diff:] = True
        else:
            collated_audios[i], audio_starts[i] = self.crop_to_max_size(
                audio, audio_size, audio_starts[i] if start_known else None
            )
    if len(audios[0].shape) == 2:
        collated_audios = collated_audios.transpose(1, 2)  # [B, T, F] -> [B, F, T]
    else:
        collated_audios = collated_audios.permute(
            (0, 4, 1, 2, 3)
        ).contiguous()  # [B, T, H, W, C] -> [B, C, T, H, W]
    return collated_audios, padding_mask, audio_starts
