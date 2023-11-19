import os

import pandas as pd
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
    def __init__(self, dataset_type, transform=None):
        if dataset_type not in ["train", "test1", "test2", "test3"]:
            raise ValueError("Invalid dataset type")
        
        if not os.path.exists(CSV_PATH):
            print("did not find csv file..")
            print("creating csv file..")
            DatasetCreater()
        
        self.df = self.get_df(dataset_type)
            
    def get_df(self, dataset_type):
        df = pd.read_csv(CSV_PATH)
        return df[df["group"] == dataset_type]
        

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # implement audio and visual
        pass
