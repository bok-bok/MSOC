import json
import os
import sys

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional
import torchaudio
import torchvision.transforms as T
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))

sys.path.append(src_path)
sys.path.append(os.path.join(src_path, "models/av_hubert/avhubert"))
# sys.path.append(os.path.join(src_path, "models/av_hubert"))

import logging

import utils as avhubert_utils

from data.augmentation import *
from util.constants import FAKE, ORIGINAL

logging.getLogger("PIL.PngImagePlugin").setLevel(logging.CRITICAL + 1)


def pil_loader(path):
    with open(path, "rb") as f:
        img = Image.open(f)
        return img.convert("RGB")


class FakeAVDataset(Dataset):
    def __init__(
        self,
        dataset_type,
        frame_num,
        classifier_type,
    ):
        # load data
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

    def _load_data(self, classifier_type: str, dataset_type: str):
        """
        Load and preprocess data from csv file
        Args:
            classifier_type (str): V, A, AV, ALL
            dataset_type (str): train, test
        """

        # check input values
        if classifier_type not in ["V", "A", "AV", "ALL"]:
            raise ValueError("classifier_type should be one of V, A, AV")
        if dataset_type not in ["train", "validation", "test"]:
            raise ValueError("dataset_type should be one of train, test")

        def get_labeled_df(classifier_type: str, df: pd.DataFrame):
            if classifier_type == "V":
                fakeset = ["FakeVideo-RealAudio", "FakeVideo-FakeAudio"]
            elif classifier_type == "A":
                fakeset = ["RealVideo-FakeAudio", "FakeVideo-FakeAudio"]
            elif classifier_type == "AV":
                fakeset = ["FakeVideo-RealAudio", "RealVideo-FakeAudio", "FakeVideo-FakeAudio"]

            df["label"] = df["type"].apply(lambda x: FAKE if x in fakeset else ORIGINAL)
            return df

        self.csv_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), "data.csv")
        self.df = pd.read_csv(self.csv_path)

        # set 70 sources for test
        self.test_source = self.df["source"].unique()[:70]

        self.df = get_labeled_df(classifier_type, self.df)
        if dataset_type == "test":
            self.df = self.df[self.df["source"].isin(self.test_source)]
        else:
            tmp_df = self.df[~self.df["source"].isin(self.test_source)]
            train_df, val_df = train_test_split(tmp_df, test_size=0.2, stratify=tmp_df["label"])
            if dataset_type == "train":
                self.df = train_df
            elif dataset_type == "val":
                self.df = val_df

        # retrict dataset to 1000 samples
        # self.df = self.df[:100]

        fake_count = len(self.df[self.df["label"] == FAKE])
        real_count = len(self.df[self.df["label"] == ORIGINAL])

        # print dataset config and label count
        print(f"Dataset config : {classifier_type}, {dataset_type}")
        print(f"fake: {fake_count}, real: {real_count}")

        # set label based on classifier type

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
        category = row["category"]
        if category == "A":
            label = ORIGINAL
        else:
            label = FAKE

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


def preemphasis(signal, coeff=0.97):
    """perform preemphasis on the input signal.

    :param signal: The signal to filter.
    :param coeff: The preemphasis coefficient. 0 is none, default 0.97.
    :returns: the filtered signal.
    """
    return np.append(signal[0], signal[1:] - coeff * signal[:-1])


class AudiosetDataset(Dataset):
    def __init__(self, dataset_json_file, label_csv=None):
        """
        Dataset that manages audio recordings
        :param audio_conf: Dictionary containing the audio loading and preprocessing settings
        :param dataset_json_file
        """
        self.datapath = dataset_json_file
        with open(dataset_json_file, "r") as fp:
            data_json = json.load(fp)

        self.data = data_json["data"]

        audio_conf = {
            "num_mel_bins": 128,
            "target_length": 1024,
            "freqm": 0,
            "timem": 0,
            "mode": "train",
            "mean": 4.2677393,
            "std": 4.5689974,
            "noise": False,
        }
        # val_audio_conf = {'num_mel_bins': 128, 'target_length': args.audio_length, 'freqm': 0, 'timem': 0, 'mixup': 0, 'dataset': args.dataset, 'mode':'evaluation', 'mean':args.dataset_mean, 'std':args.dataset_std, 'noise':False}

        self.audio_conf = {
            "num_mel_bins": 80,
        }
        print("---------------the {:s} dataloader---------------".format(self.audio_conf.get("mode")))
        self.melbins = self.audio_conf.get("num_mel_bins")
        self.freqm = self.audio_conf.get("freqm")
        self.timem = self.audio_conf.get("timem")
        print(
            "now using following mask: {:d} freq, {:d} time".format(
                self.audio_conf.get("freqm"), self.audio_conf.get("timem")
            )
        )
        # dataset spectrogram mean and std, used to normalize the input
        self.norm_mean = self.audio_conf.get("mean")
        self.norm_std = self.audio_conf.get("std")
        # if add noise for data augmentation
        self.noise = self.audio_conf.get("noise")
        if self.noise == True:
            print("now use noise augmentation")

        # self.index_dict = make_index_dict(label_csv)

        self.mel_spectrogram_transform = T.MelSpectrogram(
            sample_rate=16000,
            n_fft=512,
            win_length=int(16000 * 0.025),
            hop_length=int(16000 * 0.01),
            center=False,
            n_mels=80,
        )

    def _wav2fbank(self, filename):
        waveform, sr = torchaudio.load(filename)
        waveform = waveform - waveform.mean()

        target_length = self.audio_conf.get("target_length")
        n_frames = fbank.shape[0]

        p = target_length - n_frames

        # cut and pad
        if p > 0:
            m = torch.nn.ZeroPad2d((0, 0, 0, p))
            fbank = m(fbank)
        elif p < 0:
            fbank = fbank[0:target_length, :]

        return fbank

    def __getitem__(self, index):
        """
        returns: image, audio, nframes
        where image is a FloatTensor of size (3, H, W)
        audio is a FloatTensor of size (N_freq, N_frames) for spectrogram, or (N_frames) for waveform
        nframes is an integer
        """
        # do mix-up for this sample (controlled by the given mixup rate)
        datum = self.data[index]
        label_indices = np.zeros(self.label_num)
        fbank, mix_lambda = self._wav2fbank(datum["wav"])

        for label_str in datum["labels"].split(","):
            label_indices[int(self.index_dict[label_str])] = 1.0

        label_indices = torch.FloatTensor(label_indices)

        # SpecAug, not do for eval set
        freqm = torchaudio.transforms.FrequencyMasking(self.freqm)
        timem = torchaudio.transforms.TimeMasking(self.timem)
        fbank = torch.transpose(fbank, 0, 1)
        # this is just to satisfy new torchaudio version, which only accept [1, freq, time]
        fbank = fbank.unsqueeze(0)
        if self.freqm != 0:
            fbank = freqm(fbank)
        if self.timem != 0:
            fbank = timem(fbank)
        # squeeze it back, it is just a trick to satisfy new torchaudio version
        fbank = fbank.squeeze(0)
        fbank = torch.transpose(fbank, 0, 1)

        # normalize the input for both training and test
        if not self.skip_norm:
            fbank = (fbank - self.norm_mean) / (self.norm_std * 2)
        # skip normalization the input if you are trying to get the normalization stats.
        else:
            pass

        if self.noise == True:
            fbank = fbank + torch.rand(fbank.shape[0], fbank.shape[1]) * np.random.rand() / 10
            fbank = torch.roll(fbank, np.random.randint(-10, 10), 0)

        mix_ratio = min(mix_lambda, 1 - mix_lambda) / max(mix_lambda, 1 - mix_lambda)

        # the output fbank shape is [time_frame_num, frequency_bins], e.g., [1024, 128]
        return fbank, label_indices

    def __len__(self):
        return len(self.data)


if __name__ == "__main__":
    classifier_types = ["V", "A", "AV"]
    dataset_types = ["train", "test"]

    for classifier_type in classifier_types:
        for dataset_type in dataset_types:
            dataset = FakeAVDataset(30, classifier_type, dataset_type)
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
