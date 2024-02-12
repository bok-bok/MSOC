import json
import os
import sys

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional
import torchaudio
import torchaudio.transforms as A_T
import torchvision.transforms as T
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))

sys.path.append(src_path)
sys.path.append(os.path.join(src_path, "models/av_hubert/avhubert"))

import logging

import utils as avhubert_utils

from data.augmentation import *
from util.constants import FAKE, ORIGINAL

logging.getLogger("PIL.PngImagePlugin").setLevel(logging.CRITICAL + 1)


def pil_loader(path):
    with open(path, "rb") as f:
        img = Image.open(f)
        return img.convert("RGB")


class FakeAVDataset_new(Dataset):
    def __init__(
        self,
        dataset_type,
        frame_num,
        classifier_type,
    ):
        # load data
        self.seed = 42
        self.dataset_type = dataset_type

        self._load_data(classifier_type, dataset_type)
        self.classifier_type = classifier_type
        # set target frame number
        self.video_target_frames = frame_num
        self.audio_target_frames = frame_num * 4
        self.mel_spectrogram_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=16000,
            n_fft=512,
            win_length=int(16000 * 0.025),
            hop_length=int(16000 * 0.01),
            center=False,
            n_mels=26,
        )

        self.video_transform = transforms.Compose([Scale(size=(224, 224)), ToTensor(), Normalize()])

        lip_crop_size = 88
        lip_mean = 0.421
        lip_std = 0.165
        self.lip_transform = avhubert_utils.Compose(
            [
                avhubert_utils.Normalize(0.0, 255.0),
                avhubert_utils.CenterCrop((lip_crop_size, lip_crop_size)),
                avhubert_utils.Normalize(lip_mean, lip_std),
            ]
        )

    def get_df(self):
        return self.df

    def _load_data(self, classifier_type: str, dataset_type: str):
        """
        Load and preprocess data from csv file
        Args:
            classifier_type (str): V, A, AV, ALL
            dataset_type (str): train, test
        """

        # check input values
        if classifier_type not in ["V", "A", "AV", "ALL", "BOTH"]:
            raise ValueError("classifier_type should be one of V, A, AV")
        if dataset_type not in ["train", "val", "test", "test1", "test2"]:
            raise ValueError("dataset_type should be one of train, test")

        def get_labeled_df(classifier_type: str, df: pd.DataFrame):
            df = df.copy()
            if classifier_type == "V":
                fakeset = ["FakeVideo-RealAudio", "FakeVideo-FakeAudio"]
            elif classifier_type == "A":
                fakeset = ["RealVideo-FakeAudio", "FakeVideo-FakeAudio"]
                # remove added data - remove row contains nan
                df = df[df["target1"].notna()]

            elif classifier_type == "AV" or classifier_type == "BOTH":
                fakeset = ["FakeVideo-RealAudio", "RealVideo-FakeAudio", "FakeVideo-FakeAudio"]

            df["label"] = df["type"].apply(lambda x: FAKE if x in fakeset else ORIGINAL)
            return df

        def change_dir(df):
            def change_dir_helper(row):
                # change /neil/storage ... to /data/kyungbok
                # write code here
                cur_dir = row["preprocessed_directory"]
                new_dir = cur_dir.replace("storage/neil", "/data/kyungbok")
                return new_dir

            df["preprocessed_directory"] = df.apply(change_dir_helper, axis=1)
            return df

        self.csv_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), "final_data.csv")
        self.df = pd.read_csv(self.csv_path)
        self.df = change_dir(self.df)

        method_column_mapping = {
            "wav2lip": "source",
            "rtvc": "source",
            "faceswap": "target1",
            "fsgan": "target1",
            "faceswap-wav2lip": "target1",
            "fsgan-wav2lip": "target1",
            "real": "source",
        }
        self.df["subject"] = self.df.apply(lambda row: row[method_column_mapping[row["method"]]], axis=1)

        # set 70 sources for test
        self.test_source = self.df["subject"].unique()[:70]
        self.val_source = self.df["subject"].unique()[70:140]
        self.train_source = self.df["subject"].unique()[140:]

        self.df = get_labeled_df(classifier_type, self.df)
        if "test" in dataset_type:
            if dataset_type == "test":
                self.df = self.df[self.df["subject"].isin(self.test_source)]
            else:
                self.df = self.create_test_dataset(dataset_type)
        elif dataset_type == "train":
            self.df = self.df[self.df["subject"].isin(self.train_source)]
        elif dataset_type == "val":
            self.df = self.df[self.df["subject"].isin(self.val_source)]

        # else:
        #     tmp_df = self.df[~self.df["subject"].isin(self.test_source)]
        #     # train_df, val_df = train_test_split(
        #     #     tmp_df, test_size=0.2, stratify=tmp_df["label"], random_state=self.seed
        #     # )
        #     if dataset_type == "train":
        #         self.df = tmp_df[tmp_df["subject"].isin(self.train_source)]
        #         # self.df = train_df
        #     elif dataset_type == "val":
        #         self.df = tmp_df[tmp_df["subject"].isin(self.val_source)]
        #         # self.df = val_df

        # retrict dataset to 1000 samples
        # self.df = self.df[:100]

        fake_count = len(self.df[self.df["label"] == FAKE])
        real_count = len(self.df[self.df["label"] == ORIGINAL])

        # print dataset config and label count
        print(f"Dataset config : {classifier_type}, {dataset_type}")
        print(f"fake: {fake_count}, real: {real_count}")

        # set label based on classifier type

    def create_test_dataset(self, test_type: str):
        df = self.df[self.df["source"].isin(self.test_source)]
        # Create a method balanced dataset
        if test_type == "test1":
            total_sample_size = 70
            methods = list(df["method"].unique())
            methods.remove("real")

            num_methods = len(methods)
            # Step 2: Calculate sample size per method
            sample_size_per_method = total_sample_size // num_methods

            # Step 3: Handle the remainder
            remainder = total_sample_size % num_methods

            real = df[df["method"] == "real"]

            # Step 4: Sample from each method group
            samples = []
            for method in methods:
                method_sample_size = sample_size_per_method + (1 if remainder > 0 else 0)
                remainder -= 1
                method_samples = df[df["method"] == method].sample(
                    n=method_sample_size, replace=False, random_state=self.seed
                )
                samples.append(method_samples)

            # add real samples
            samples.append(real)
            # Step 5: Combine the samples
            method_balanced_df = pd.concat(samples)
            return method_balanced_df

        elif test_type == "test2":
            total_sample_size = 70
            categories = list(df["type"].unique())
            categories.remove("RealVideo-RealAudio")

            sample_size_per_category = total_sample_size // len(categories)
            # Step 3: Handle the remainder
            remainder = total_sample_size % len(categories)
            real = df[df["method"] == "real"]

            samples = []
            for category in categories:
                category_sample_size = sample_size_per_category + (1 if remainder > 0 else 0)
                remainder -= 1

                category_samples = df[df["type"] == category].sample(
                    n=category_sample_size, replace=False, random_state=self.seed
                )
                samples.append(category_samples)
            samples.append(real)
            # Step 5: Combine the samples
            test_2_df = pd.concat(samples)
            return test_2_df

    def stacker(self, feats, stack_order):
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

    def __getitem__(self, index):
        row = self.df.iloc[index]

        # get label
        # category = row["category"]
        # if category == "A":
        #     label = ORIGINAL
        # else:
        #     label = FAKE
        label = row["label"]

        directory = row["preprocessed_directory"]
        if self.classifier_type == "V":
            frames = self._load_frames(directory)  # [B, T, C, H, W]
            return frames, label

        elif self.classifier_type == "AV":
            lip_frames = self._load_lip_frames(directory)  # [B, C, T, H, W]
            spectrogram = self._load_audio(directory)  # [B ,T, F]
            stacked_spectrogram = self.stacker(spectrogram, 4)  # [B, T, F]
            stacked_spectrogram = stacked_spectrogram.transpose(1, 0)  # [B, F, T]
            return lip_frames, stacked_spectrogram, label
        elif self.classifier_type == "A":
            spectrogram = self._load_audio(directory)
            return spectrogram, label
        elif self.classifier_type == "BOTH":
            frames = self._load_frames(directory)
            spectrogram = self._load_audio(directory)
            return frames, spectrogram, label
        elif self.classifier_type == "ALL":
            frames = self._load_frames(directory)
            lip_frames = self._load_lip_frames(directory)
            spectrogram = self._load_audio(directory)
            stacked_spectrogram = self.stacker(spectrogram, 4)
            stacked_spectrogram = stacked_spectrogram.transpose(1, 0)
            return frames, lip_frames, stacked_spectrogram, label

        # return spectrogram, label
        return frames, lip_frames, spectrogram, stacked_spectrogram, label

    def _load_frames(self, directory: str):
        """
        Load frames from directory
        Args:
            directory (str): directory path

        Returns:
            t_seq: torch.tensor, [B, T, C, H, W]
        """
        # load fixed number of frames
        frame_dir = os.path.join(directory, "frames")

        # frame_0.jpg -> 0
        sorted_frames_files = sorted(os.listdir(frame_dir), key=lambda x: int(x.split(".")[0].split("_")[1]))
        frame_num = len(sorted_frames_files)

        # clip frames to target number
        if frame_num > self.video_target_frames:
            sorted_frames_files = sorted_frames_files[: self.video_target_frames]

        # load and transform frames
        seq = [pil_loader(os.path.join(frame_dir, img)) for img in sorted_frames_files]
        t_seq = self.video_transform(seq)
        t_seq = torch.stack(t_seq, 0)

        # pad frames to target number
        if frame_num < self.video_target_frames:
            p = self.video_target_frames - frame_num
            t_seq = torch.nn.functional.pad(t_seq, (0, 0, 0, 0, 0, 0, 0, p))
        return t_seq

    def _load_lip_frames(self, directory: str):
        """
        Load lip frames from directory
        Args:
            directory (str): directory path

        Returns:
            t_seq: torch.tensor, [B, C, T, H, W]
        """

        def pil_loader_gray(path):
            with open(path, "rb") as f:
                img = Image.open(f)
                img = np.array(img)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                return img

        # load fixed number of frames
        lip_frame_dir = os.path.join(directory, "lip")

        # frame_0.jpg -> 0
        sorted_frames_files = sorted(
            os.listdir(lip_frame_dir), key=lambda x: int(x.split(".")[0].split("_")[1])
        )
        frame_num = len(sorted_frames_files)

        # clip frames to target number
        if frame_num > self.video_target_frames:
            sorted_frames_files = sorted_frames_files[: self.video_target_frames]

        # load and transform frames
        seq = [pil_loader_gray(os.path.join(lip_frame_dir, img)) for img in sorted_frames_files]
        seq = np.stack(seq)
        t_seq = self.lip_transform(seq)
        t_seq = torch.tensor(t_seq, dtype=torch.float32).unsqueeze(0)

        # pad frames to target number
        if frame_num < self.video_target_frames:
            p = self.video_target_frames - frame_num
            t_seq = torch.nn.functional.pad(t_seq, (0, 0, 0, 0, 0, 0, 0, p))

        return t_seq

    def _load_audio(self, audio_dir: str):
        # load audio
        audio_dir = os.path.join(audio_dir, "audio.wav")
        waveform, sr = torchaudio.load(audio_dir)
        waveform = waveform - waveform.mean()

        # apply random augmentation
        if self.dataset_type == "train":
            waveform = add_noise(waveform)

        mel_spectrogram = self.mel_spectrogram_transform(waveform[0])

        # pad or truncate to target length
        n_frames = mel_spectrogram.shape[1]
        p = self.audio_target_frames - n_frames
        if p > 0:
            # m = torch.nn.ZeroPad2d((0, 0, 0, p))
            m = torch.nn.ZeroPad2d((0, p, 0, 0))
            mel_spectrogram = m(mel_spectrogram)
        elif p < 0:
            mel_spectrogram = mel_spectrogram[:, 0 : self.audio_target_frames]

        # [F, T] => [T, F]
        mel_spectrogram = mel_spectrogram.transpose(1, 0)
        # mel_spectrogram = mel_spectrogram.numpy().transpose(1, 0)
        # # stack audio frames to better align with video frames
        # feats = stacker(mel_spectrogram, 4)

        # feats = feats.transpose(1, 0)
        # print(f"feat shape: {feats.shape}")

        return mel_spectrogram

    def __len__(self):
        return len(self.df)


def add_noise(audio, min_noise_level=0.005, max_noise_level=0.05):
    noise_level = random.uniform(min_noise_level, max_noise_level)
    noise = torch.randn(audio.shape) * noise_level
    noisy_audio = audio + noise
    return noisy_audio


if __name__ == "__main__":
    classifier_types = ["V"]
    dataset_types = ["train", "val"]

    for classifier_type in classifier_types:
        for dataset_type in dataset_types:
            dataset = FakeAVDataset_new(
                dataset_type,
                16,
                classifier_type,
            )
    # result
    # Dataset config : V, train
    # fake: 17674, real: 860

    # Dataset config : V, test
    # fake: 2892, real: 140

    # Dataset config : A, train
    # fake: 9767, real: 8767

    # Dataset config : A, test
    # fake: 1590, real: 1442

    # Dataset config : AV, train
    # fake: 18104, real: 430

    # Dataset config : AV, test
    # fake: 2962, real: 70
