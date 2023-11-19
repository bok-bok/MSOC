import os

import pandas as pd

from utils.constants import (
    AUDIO_MODELS_TRAIN,
    TRAIN_GROUP,
    VISUAL_BLENDINGS_TRAIN,
    VISUAL_MODELS_TRAIN,
)

SWAN_DIR = "/storage/neil/SWAN-DF/SWAN-DF/"


class DatasetCreater:
    
    def __init__(self):
        self.dataset_dir = SWAN_DIR
        self.get_files()
        self.create_df()

    def create_df(self):
        # print(len(self.visual.keys()))
        def check_valid(video_row, audio_row):
            if video_row["subject1"] != audio_row["subject1"] or video_row["subject2"] != audio_row["subject2"]:
                raise ValueError("subject1 and subject2 are not the same")
        data = []
        # iterate through all the keys
        for key in self.keys:
            for _, video_row in self.visual_df[self.visual_df["key"] == key].iterrows():
                for _,audio_row in self.audio_df[self.audio_df["key"] == key].iterrows():
                    check_valid(video_row, audio_row)
                    video_model = video_row["model_name"]
                    blending_name = video_row["blending_name"]
                    audio_model = audio_row["model_name"]
                    pair = f"{video_row['subject1']}_{video_row['subject2']}"
                    
                    group = self.check_group(video_model, blending_name, audio_model, pair)


                    data.append({
                        "video_model": video_model,
                        "blending_name": blending_name,
                        "audio_model": audio_model,
                        "pair": pair,
                        "group": group,

                        "video_dir": video_row["file_dir"],
                        "audio_dir": audio_row["file_dir"],
                        "target" : 1
                    })

        self.df = pd.DataFrame(data)
        print(f"total_count: {len(self.df)}")
        self.df.to_csv("./utils/swan_df.csv")
                    
    def check_group(self, video_model, blending_name, audio_model, pair):
        # train
        # seen visual   & seen audio model

        # test1
        # seen visual   & unseen audio model

        # test2
        # unseen visual & seen audio model

        # test3 
        # unseen visual & sudio model

        # check include in train or not 
        train_audio = False
        train_visual = False
        train_group = False
        if video_model in VISUAL_MODELS_TRAIN and blending_name in VISUAL_BLENDINGS_TRAIN:
            train_visual = True
        if audio_model in AUDIO_MODELS_TRAIN:
            train_audio = True
        if pair in TRAIN_GROUP:
            train_group = True

        
        if train_visual and train_audio and train_group:
            return "train"
        elif train_visual and not train_audio and not train_group:
            return "test1"
        elif not train_visual and train_audio and not train_group:
            return "test2"
        elif not train_visual and not train_audio and not train_group:
            return "test3"
        else:
            return "drop"

        
        
    def get_dir_info(self, dir):
        dir_info = dir.split("/")
        target = dir_info[-2]

        # dir_info[-1] is the subject id 1
        if target in ["160x160", "256x256", "320x320"]:
            return target, dir_info[-1]
        elif target == "wav":
            return target, dir_info[-1]
        else:
            return None, None
    

    def get_visual_modal_info(self, file_name:str):
        file_split = file_name.split("-")
        video_name = file_split[0]
        model_name = file_split[1].split("_")[1] +"_" +  file_split[2]
        bledning_name = file_split[3].split("_")[1]
        subject2 = file_split[-1].split(".")[0]
        key = video_name+"_"+subject2

        return key, subject2, video_name, model_name, bledning_name


    def get_aduio_modal_info(self, file_name:str):
        
        file_info = file_name.split('-')
        model_name = file_info[1] +"-" +  file_info[2]+"-"+  file_info[3]
        subject2 = file_info[-1].split(".")[0]
        audio_name = file_info[0]
        subject2 = file_info[-1].split(".")[0]
        key = audio_name+"_"+subject2

        return key, subject2, audio_name, model_name


    def get_files(self):
        visual_data = []
        audio_data = []

        # iterate through all the files
        for root, _ ,filenames in os.walk(self.dataset_dir):
            # split get information
            modal, subject1 = self.get_dir_info(root)  
            if modal is not None:
                # audio
                if modal =="wav":
                    for filename in filenames:
                        key, subject2, audio_name, model_name = self.get_aduio_modal_info(filename)
                        # id = subject + "-" + subject2
                        file_dir = os.path.join(root, filename)
                        audio_data.append(
                            {
                            "key": key,
                            "subject1": subject1,
                            "subject2": subject2,
                            "audio_name": audio_name,
                            "model_name": model_name,
                            "file_dir": file_dir,
                            }
                        )
                # visual
                else:
                    for filename in filenames:
                        key, subject2, video_name, model_name, blending_name = self.get_visual_modal_info(filename)
                        # id = subject + "-" + subject2
                        file_dir = os.path.join(root, filename)
                        visual_data.append(
                            {
                            "key": key,
                            "subject1": subject1,
                            "subject2": subject2,
                            "video_name": video_name,
                            "model_name": model_name,
                            "blending_name": blending_name,
                            "file_dir": file_dir,
                            }
                        )
        
        # create audio and visual df
        self.audio_df = pd.DataFrame(audio_data)
        self.visual_df = pd.DataFrame(visual_data)

        # get keys
        self.keys = self.visual_df["key"].unique()



if __name__ == "__main__":
    DatasetCreater()