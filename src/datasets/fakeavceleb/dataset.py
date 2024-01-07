import json
import os
import sys
from multiprocessing import Pool

import numpy as np
import pandas as pd
import torch
import torch.nn.functional
import torchaudio
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import Dataset

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from data.augmentation import *
from utils.constants import FAKE, ORIGINAL


def pil_loader(path):
    with open(path, "rb") as f:
        img = Image.open(f)
        return img.convert("RGB")


class FakeAVDataset(Dataset):
    def __init__(self, frame_num, classifier_type, dataset_type):
        # load data
        self._load_data(classifier_type, dataset_type)

        # set target frame number
        self.video_target_frames = frame_num
        self.audio_target_frames = frame_num * 4
        self.mel_spectrogram_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=16000,
            n_fft=512,
            win_length=int(16000 * 0.025),
            hop_length=int(16000 * 0.01),
            center=False,
            n_mels=80,
        )

        self.video_transform = transforms.Compose([Scale(size=(224, 224)), ToTensor(), Normalize()])

    def _load_data(self, classifier_type: str, dataset_type: str):
        """
        Load and preprocess data from csv file
        Args:
            classifier_type (str): V, A, AV
            dataset_type (str): train, test
        """

        # check input values
        if classifier_type not in ["V", "A", "AV"]:
            raise ValueError("classifier_type should be one of V, A, AV")
        if dataset_type not in ["train", "test"]:
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

        if dataset_type == "train":
            self.df = self.df[~self.df["source"].isin(self.test_source)]
        elif dataset_type == "test":
            self.df = self.df[self.df["source"].isin(self.test_source)]

        self.df = get_labeled_df(classifier_type, self.df)

        fake_count = len(self.df[self.df["label"] == FAKE])
        real_count = len(self.df[self.df["label"] == ORIGINAL])

        # print dataset config and label count
        print(f"Dataset config : {classifier_type}, {dataset_type}")
        print(f"fake: {fake_count}, real: {real_count}")

        # set label based on classifier type

    def __getitem__(self, index):
        row = self.df.iloc[index]

        directory = row["preprocessed_directory"]

        frames = self._load_frames(directory)
        spectrogram = self._load_audio(directory)

        # get label
        category = row["category"]
        if category == "A":
            label = ORIGINAL
        else:
            label = FAKE

        # return spectrogram, label
        return frames, spectrogram, label

    def _load_frames(self, directory: str):
        # load fixed number of frames
        frame_dir = os.path.join(directory, "frames")

        # frame_0.jpg -> 0
        sorted_frames_files = sorted(
            os.listdir(frame_dir), key=lambda x: int(x.split(".")[0].split("_")[1])
        )
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
        mel_spectrogram = torch.transpose(mel_spectrogram, 0, 1)
        # [time_frame_num, frequency_bins], ex) [120, 80]
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
        print(
            "---------------the {:s} dataloader---------------".format(self.audio_conf.get("mode"))
        )
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
