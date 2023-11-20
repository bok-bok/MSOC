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
    TRAIN_GROUP,
    VISUAL_BLENDINGS_TRAIN,
    VISUAL_MODELS_TRAIN,
)

SWAN_DIR = "/storage/neil/SWAN-DF/SWAN-DF/"
CSV_PATH = "./utils/swan_df.csv"


class SwanDataset(Dataset):
    def __init__(self, dataset_type: str,  transform=None):
        if dataset_type not in ["train", "test1", "test2", "test3"]:
            raise ValueError("Invalid dataset type")

        self.target_sample_rate = 16000
        self.feat_len = 750

        self.lfcc_transform = T.LFCC(
            sample_rate = self.target_sample_rate,
            n_lfcc = 40
        )
          

        self.load_data(dataset_type=dataset_type)
            
        

    def __len__(self):
        return len(self.df)


    def __getitem__(self, idx):
        # implement audio and visual
        row = self.df.iloc[idx]
        audio_dir = row["audio_dir"]
        waveform , sample_rate  = torchaudio.load(audio_dir)

        lfcc = self.lfcc_transform(waveform)

        # adjust feature dim length
        this_lfcc_len = lfcc.shape[2]
        if this_lfcc_len > self.feat_len:
            startp = np.random.randint(this_lfcc_len - self.feat_len)
            lfcc = lfcc[:,:,  startp:startp + self.feat_len]
        if this_lfcc_len < self.feat_len:
            lfcc = padding(lfcc, self.feat_len)
        # lfcc shape 
        # [1, 40, 750]
        

        target = row["target"]
        
        return lfcc, target

    def load_data(self, dataset_type:str):
        if not os.path.exists(CSV_PATH):
            print("did not find csv file..")
            print("creating csv file..")
            DatasetCreater()

        self.df = self.get_df(dataset_type)



    def get_df(self, dataset_type:str):
        df = pd.read_csv(CSV_PATH)
        return df[df["group"] == dataset_type]

def padding(spec, ref_len):
    channel, width, cur_len = spec.shape
    assert ref_len > cur_len
    padd_len = ref_len - cur_len
    return torch.cat((spec, torch.zeros(channel, width, padd_len, dtype=spec.dtype)), 2)

def repeat_padding(spec, ref_len):
    mul = int(np.ceil(ref_len / spec.shape[1]))
    spec = spec.repeat(1, mul)[:, :ref_len]
    return spec