import multiprocessing as mp
import os
import random
from dataclasses import dataclass
from typing import List, Optional

import cv2
import numpy
import numpy as np
import pandas as pd
import python_speech_features
import torch
from pytorch_lightning import LightningDataModule
from pytorch_lightning.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
from scipy.io import wavfile
from scipy.signal import resample
from torch.utils.data import DataLoader

import models.avhubert.utils as custom_utils

# from models.faceDetector.faceCropper import FaceCropper
from new_datasets.augmentations import shift_audio, stacker
from new_datasets.dataset_utils import split_new_dataset

mp.set_start_method("spawn", force=True)
fps = 25.0


@dataclass
class Metadata:
    source: str
    target1: str
    target1: str
    method: str
    category: str
    type: str
    race: str
    gender: str
    vid: str
    path: str


dtype = {
    "source": str,
    "target1": str,
    "target2": str,
    "method": str,
    "category": str,
    "type": str,
    "race": str,
    "gender": str,
    "vid": str,
    "path": str,
}


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


g = torch.Generator()
g.manual_seed(0)


class Fakeavceleb(object):
    def __init__(
        self,
        subset: str,
        root: str = "data",
        metadata: Optional[List[Metadata]] = None,
        # augmentation: bool = False,
        stack_audio=False,
    ):
        # self.augmentation = augmentation
        self.root = root
        self.subset = subset
        self.metadata = metadata

        self.image_mean = 0.421
        self.image_std = 0.165
        self.numFrames = 30
        self.scale_percent = 0.5
        self.audio_shift = 0.2

        self.stack_order_audio = 4
        self.stack_audio = stack_audio

        print(f"stack_audio: {self.stack_audio}")

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
        meta = self.metadata.iloc[index]

        c_label = 1 if ("RealVideo" in meta.type) and ("RealAudio" in meta.type) else 0
        v_label = 1 if "RealVideo" in meta.type else 0
        a_label = 1 if "RealAudio" in meta.type else 0
        m_label = 1 if "real" in meta.method else 0

        if "RealVideo-RealAudio" in meta.type:
            mm_label = 0
        elif "FakeVideo-RealAudio" in meta.type:
            mm_label = 1
        elif "RealVideo-FakeAudio" in meta.type:
            mm_label = 2
        else:
            mm_label = 3
        if meta.category == "B":
            s_label = 0  # real_video-fake_audio
        else:
            s_label = 1

        path = "/".join(meta.path.split("/")[1:])
        file_path = os.path.join(self.root, path, meta.vid)

        file_path = file_path.replace(" (", "_")
        file_path = file_path.replace(")", "")
        video_fn = file_path
        if meta.category == "E":
            if "fake" in file_path:
                audio_fn = file_path.replace(".mp4", "_sync.wav")
            else:
                audio_fn = file_path.replace(".mp4", "_fake_sync.wav")
        else:
            audio_fn = file_path.replace(".mp4", ".wav")

        video, audio = self.load_features(video_fn, audio_fn, meta.category)

        # if modified:
        #     # assert self.augmentation
        #     s_label = 0
        #     m_label = 0

        return {
            "id": index,
            "file": os.path.join(meta.path, meta.vid),
            "video": torch.FloatTensor(video),
            "audio": torch.FloatTensor(audio),
            "padding_mask": torch.tensor([False]),
            "v_label": torch.tensor(v_label),
            "a_label": torch.tensor(a_label),
            "c_label": torch.tensor(c_label),
            "m_label": torch.tensor(m_label),
            "mm_label": torch.tensor(mm_label),
            "s_label": torch.tensor(s_label),
        }

    def __len__(self):
        return self.metadata.shape[0]

    def load_features(self, video_fn, audio_fn, category):

        audio = self.load_audio(audio_fn, category)

        video = self.load_video(video_fn)

        return video, audio

    def load_audio(self, audio_path, category):
        sample_rate, audio = wavfile.read(audio_path)
        if len(audio.shape) > 1:
            audio = audio[:, 0]

        # if category is F and subset is test, we apply augmentation
        if category == "F":
            assert self.subset in "test"
            audio = shift_audio(audio, self.audio_shift)

        if sample_rate != 16_000:
            num_samples = round(len(audio) * float(16_000) / sample_rate)
            audio = resample(audio, num_samples)
            sample_rate = 16000

        assert sample_rate == 16_000

        # fps is not always 25, in order to align the visual, we modify the window and step in MFCC extraction process based on fps
        if self.stack_audio:
            numcep = 26
        else:
            numcep = 13

        audio = python_speech_features.mfcc(
            audio, 16000, numcep=numcep, winlen=0.025 * 25 / fps, winstep=0.010 * 25 / fps
        )
        # audio = python_speech_features.logfbank(
        #     audio, 16000, winlen=0.025 * 25 / fps, winstep=0.010 * 25 / fps,
        # )

        maxAudio = int(
            self.numFrames * 4
        )  # audio frame is 10*25/fps ms long, visual frame is 1000/fps ms long

        if audio.shape[0] < maxAudio:
            shortage = maxAudio - audio.shape[0]
            audio = numpy.pad(audio, ((0, shortage), (0, 0)), "wrap")

        # audio : [T[numFrames*4], F]
        audio = audio[: int(round(self.numFrames * 4)), :]
        audio = audio.astype(numpy.float32)

        if self.stack_audio:
            audio = stacker(audio, self.stack_order_audio)
            audio = audio.transpose(1, 0)

        return audio

    def __load_video(self, path, scale_percent):
        cap = cv2.VideoCapture(path)
        frames = []
        frame_count = 0
        cropped_frames = []
        no_face = False
        previous_mask = None
        while True:
            ret, frame = cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                dim = (int(frame.shape[1] * scale_percent), int(frame.shape[0] * scale_percent))
                frame = cv2.resize(frame, dim)
                frames.append(frame)
                frame_count += 1
                if len(frames) == self.numFrames:
                    break
            else:
                break

        if frame_count < self.numFrames:
            shortage = self.numFrames - frame_count
            frame_like_zero = np.zeros((dim[1], dim[0], 3))
            for _ in range(shortage):
                frames.append(frame_like_zero)

        frames = np.stack(frames)

        return frames

    def load_video(self, video_name):
        feats = self.__load_video(video_name, self.scale_percent)

        feats = self.transform(feats)
        feats = feats.transpose(3, 0, 1, 2)
        return feats

    def load_visual(self, dataPath, aug=False):
        H = 112
        cap = cv2.VideoCapture(dataPath)
        frames = []
        frame_count = 0
        flip = False
        if aug:
            p = random.random()

            # crop
            new = int(H * random.uniform(0.7, 1))
            x, y = numpy.random.randint(0, H - new), numpy.random.randint(0, H - new)

            # flip
            if p < 0.5:
                flip = True

        while True:
            ret, frame = cap.read()

            if ret:
                frame_count += 1
                frame = cv2.resize(frame, (H, H))
                if flip:
                    frame = cv2.flip(frame, 1)
                if aug:
                    frame = cv2.resize(frame[y : y + new, x : x + new], (H, H))
                frames.append(frame)
            else:
                break

            if frame_count == self.numFrames:
                break
        cap.release()
        frames = numpy.array(frames)
        frames = frames.transpose(3, 0, 1, 2)
        return frames


class FakeavcelebDataModule(LightningDataModule):
    train_dataset: Fakeavceleb
    dev_dataset: Fakeavceleb
    test_dataset: Fakeavceleb
    metadata: List[Metadata]

    def __init__(
        self,
        root: str = "/data/kyungbok/FakeAVCeleb_v1.2",
        batch_size: int = 1,
        num_workers: int = 0,
        max_sample_size=30,
        take_train: int = None,
        take_dev: int = None,
        take_test: int = None,
        augmentation: bool = False,
        dataset_type: str = "new",
        test_subset: str = "all",
        stack_audio=False,
        # mask_face=True,
    ):
        print("batch_size", batch_size)
        print("num_workers", num_workers)
        super().__init__()
        self.root = root
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.take_train = take_train
        self.take_dev = take_dev
        self.take_test = take_test
        self.Dataset = Fakeavceleb
        self.pad_audio = False
        self.random_crop = False
        self.max_sample_size = max_sample_size
        # self.augmentation = augmentation
        self.dataset_type = dataset_type
        self.test_subset = test_subset
        self.stack_audio = stack_audio
        # self.mask_face = mask_face

    def setup(self, stage: Optional[str] = None) -> None:
        self.metadata = pd.read_csv(os.path.join(self.root, "meta_data_added.csv"), dtype=dtype)  # .loc
        self.metadata.columns = dtype.keys()
        # if self.dataset_type == "original":
        #     self.set_original_dataset()
        # elif self.dataset_type == "new":
        self.set_new_dataset()

    def set_new_dataset(self):
        print("new dataset")
        df = self.metadata

        self.train_metadata, self.val_metadata, self.test_metadata = split_new_dataset(
            df=df, test_subset=self.test_subset
        )
        self.train_dataset = self.Dataset(
            "train",
            self.root,
            metadata=self.train_metadata,
            stack_audio=self.stack_audio,
        )

        self.val_dataset = self.Dataset(
            "val",
            self.root,
            metadata=self.val_metadata,
            stack_audio=self.stack_audio,
        )
        self.test_dataset = self.Dataset(
            "test",
            self.root,
            metadata=self.test_metadata,
            stack_audio=self.stack_audio,
        )

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            worker_init_fn=seed_worker,
            generator=g,
        )

    def val_dataloader(self) -> EVAL_DATALOADERS:
        if self.dataset_type == "original":
            return DataLoader(
                self.test_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=True,
                worker_init_fn=seed_worker,
                generator=g,
            )  # , worker_init_fn=seed_worker, generator=g
        else:
            return DataLoader(
                self.val_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=True,
                worker_init_fn=seed_worker,
                generator=g,
            )

    def test_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            worker_init_fn=seed_worker,
            generator=g,
        )
