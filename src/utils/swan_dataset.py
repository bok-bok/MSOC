import os

import numpy as np
import pandas as pd
import torch
import torchaudio
import torchaudio.transforms as T
from torch.utils.data import DataLoader, Dataset

from utils import DatasetCreater
from utils.constants import (
    AUDIO_MODELS_TRAIN,
    FAKE,
    ORIGINAL,
    TRAIN_GROUP,
    VISUAL_BLENDINGS_TRAIN,
    VISUAL_MODELS_TRAIN,
)

SWAN_DIR = "/storage/neil/SWAN-DF/SWAN-DF/"
CSV_PATH = "./utils/swan_df.csv"


class SwanDataset(Dataset):
    def __init__(self, dataset_type: str, transform=None):
        if dataset_type not in ["train", "test1", "test2", "test3"]:
            raise ValueError("Invalid dataset type")

        self.target_sample_rate = 16000
        self.feat_len = 750
        self.random_seed = 42

        self.lfcc_transform = torchaudio.transforms.LFCC(
            sample_rate=16000,
            n_filter=20,
            f_min=0.0,
            f_max=None,
            n_lfcc=60,  # Assuming you want the same number of coefficients as filters
            dct_type=2,
            norm="ortho",
            log_lf=False,
            speckwargs={"n_fft": 512, "win_length": 320, "hop_length": 160, "center": False},
        )

        # self.lfcc_transform = T.LFCC(
        #     sample_rate = self.target_sample_rate,
        #     n_lfcc = 40
        # )

        self.load_data(dataset_type=dataset_type)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # implement audio and visual
        row = self.df.iloc[idx]
        target = row["target"]

        if target == FAKE:
            return self.__get_fake_item__(row)
        else:
            return self.__get_real_item__(row)

    def __get_balance_dataset__(self, df):
        """
        Balance the dataset by sampling 2x of real count from fake and retain all real
        """
        real_count = len(df[df["target"] == ORIGINAL])

        # sample only 2x of real count from fake and retain all real
        fake_df = df[df["target"] == FAKE].sample(n=real_count * 2, random_state=self.random_seed)
        real_df = df[df["target"] == ORIGINAL]
        print(f"fake: {len(fake_df)} real: {len(real_df)}")
        df = pd.concat([fake_df, real_df])
        # shuffle
        df = df.sample(frac=1, random_state=self.random_seed).reset_index(drop=True)
        return df

    def __get_fake_item__(self, row):
        audio_dir = row["audio_dir"]
        video_dir = row["video_dir"]
        waveform, sample_rate = torchaudio.load(audio_dir)

        lfcc = self.__process_lfcc__(waveform)

        target = row["target"]

        return lfcc, target

    def __get_real_item__(self, row):
        vidio_dir = row["video_dir"]
        waveform, sample_rate = torchaudio.load(vidio_dir)

        lfcc = self.__process_lfcc__(waveform)

        return lfcc, row["target"]

    def __process_lfcc__(self, waveform):
        lfcc = self.lfcc_transform(waveform)

        # adjust feature dim length
        this_lfcc_len = lfcc.shape[2]
        if this_lfcc_len > self.feat_len:
            startp = np.random.randint(this_lfcc_len - self.feat_len)
            lfcc = lfcc[:, :, startp : startp + self.feat_len]
        if this_lfcc_len < self.feat_len:
            lfcc = padding(lfcc, self.feat_len)
        return lfcc

    def load_data(self, dataset_type: str):
        if not os.path.exists(CSV_PATH):
            print("did not find csv file..")
            print("creating csv file..")
            DatasetCreater()

        self.df = self.get_df(dataset_type)
        # get only 1000

    def get_df(self, dataset_type: str):
        df = pd.read_csv(CSV_PATH)
        df = df[df["group"] == dataset_type]
        df = self.__get_balance_dataset__(df)
        return df


def padding(spec, ref_len):
    channel, width, cur_len = spec.shape
    assert ref_len > cur_len
    padd_len = ref_len - cur_len
    return torch.cat((spec, torch.zeros(channel, width, padd_len, dtype=spec.dtype)), 2)


def repeat_padding(spec, ref_len):
    mul = int(np.ceil(ref_len / spec.shape[1]))
    spec = spec.repeat(1, mul)[:, :ref_len]
    return spec
