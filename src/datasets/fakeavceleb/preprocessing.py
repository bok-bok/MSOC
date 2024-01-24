import os
import sys

import cv2
import moviepy.editor as mp
import pandas as pd
import torch
import torchaudio
import torchaudio.transforms as T

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from data.detect_landmark import preprocess_frames

output_path = "/storage/neil/FakeAVCeleb_preprocessed"


data_csv_path = "/storage/neil/FakeAVCeleb_v1.2/meta_data.csv"
data_path = "/storage/neil/FakeAVCeleb_v1.2"
df = pd.read_csv(data_csv_path)


def create_dir(row):
    cur_path = row["Unnamed: 9"]
    # remove first path
    new_path = "/".join(cur_path.split("/")[1:])
    return os.path.join(new_path, row["path"])


df["directory"] = df.apply(create_dir, axis=1)


def extract_all_frames(video_path, frames_dir):
    os.makedirs(frames_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_file = os.path.join(frames_dir, f"frame_{frame_count}.jpg")
        cv2.imwrite(frame_file, frame)
        frame_count += 1

    cap.release()


def extract_all_lips(base_path):
    face_predictor_path = "models/preprocessing_model_weights/shape_predictor_68_face_landmarks.dat"
    mean_face_path = "models/preprocessing_model_weights/20words_mean_face.npy"
    preprocess_frames(base_path, face_predictor_path, mean_face_path)


def extract_audio(video_path, audio_dir, sample_rate=16000):
    video = mp.VideoFileClip(video_path)
    audio_path = os.path.join(audio_dir, "audio.wav")
    video.audio.write_audiofile(audio_path, fps=sample_rate)


sample_rate = 16000
mel_spectrogram_transform = T.MelSpectrogram(
    sample_rate=sample_rate,
    n_fft=512,
    win_length=int(sample_rate * 0.025),
    hop_length=int(sample_rate * 0.01),
    center=False,
    n_mels=80,
)


def extract_mel_spectrogram(audio_path):
    waveform, sr = torchaudio.load(audio_path)
    mel_spec = mel_spectrogram_transform(waveform)
    return mel_spec


# extract frames and audio(.wav) from videos
preprocessed_data_directories = []
for idx, row in df.iterrows():
    # create and store output_data_dir
    directory = row["directory"]
    data_dir = os.path.join(data_path, directory)
    out_data_dir = os.path.join(output_path, directory[:-4])
    preprocessed_data_directories.append(out_data_dir)

    frames_dir = os.path.join(out_data_dir, "frames")

    # extract frames and audio
    # extract_all_frames(data_dir, frames_dir)
    # extract_audio(data_dir, out_data_dir)
    # extract lips
    extract_all_lips(out_data_dir)
# create preprocessed_directory column
df["preprocessed_directory"] = preprocessed_data_directories


# extract mel spectrograms from audio
for idx, row in df.iterrows():
    path = row["preprocessed_directory"]
    audio_path = os.path.join(path, "audio.wav")
    mel_spectrogram = extract_mel_spectrogram(audio_path=audio_path)
    save_path = os.path.join(path, "mel_spectrogram.pt")
    print(f"Saving mel spectrogram {mel_spectrogram.shape} for {save_path}")
    torch.save(mel_spectrogram, save_path)

# save dataframe with preprocessed_directory column
save_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "data.csv"))
df.to_csv(save_path, index=False)
