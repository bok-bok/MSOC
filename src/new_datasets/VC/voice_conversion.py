from TTS.api import TTS
import pandas as pd
import torch


device = "cuda" if torch.cuda.is_available() else "cpu"

# Load the model
tts = TTS(model_name="voice_conversion_models/multilingual/vctk/freevc24", progress_bar=True).to(device)

df = pd.read_csv("paired_fakeaudio_dir.csv")

for i in range(len(df)):
    source_wav_path = df.iloc[i, 0]
    target_wav_path = df.iloc[i, 1]
    output_wav_path = df.iloc[i, 2]

    tts.voice_conversion_to_file(source_wav=source_wav_path, \
                                target_wav=target_wav_path, 
                                file_path=output_wav_path)


