import os
import sys

# import fairseq
# import hubert
# import hubert_asr
# import hubert_pretraining
import torch

from models.ast_models import ASTModel

# from models.av_model import AVModel
from models.vivit import ViViT
from util.constants import (
    HUBERT_ENCODER_MODEL_STATE_DICT_PATH,
    MC_TCN_MODEL_PATH,
    MODEL_WEIGHTS_PATH,
)

# sys.path.append("./av_hubert")


# def load_av_model(pretrained: bool, device: str):
#     model = AVModel(pretrained=pretrained, device=device)
#     return model


def load_ast(label_dim: int, input_fdim: int, input_tdim: int, pretrained: bool, device: str):
    if pretrained:
        model_name = f"ast_model_{label_dim}_{input_fdim}_{input_tdim}.pth"
        model_path = os.path.join(MODEL_WEIGHTS_PATH, model_name)
        print(f"Loading AST model from {model_path}")
        if not os.path.exists(model_path):
            raise Exception(f"Model {model_name} does not exist")
        model = torch.load(model_path)

        return model.to(device)
    else:
        return ASTModel(label_dim=label_dim, input_fdim=input_fdim, input_tdim=input_tdim).to(device)


def load_vivit(
    image_size: int,
    patch_size: int,
    num_classes: int,
    num_frames: int,
    pretrained: bool,
    device: str,
):
    if pretrained:
        model_name = f"vivit_model_{image_size}_{patch_size}_{num_classes}_{num_frames}.pth"
        model_path = os.path.join(MODEL_WEIGHTS_PATH, model_name)
        print(f"Loading ViViT model from {model_path}")
        if not os.path.exists(model_path):
            raise Exception(f"Model {model_name} does not exist")
        model = torch.load(model_path)
        return model.to(device)
    else:
        return ViViT(
            image_size=image_size,
            patch_size=patch_size,
            num_classes=num_classes,
            num_frames=num_frames,
        ).to(device)
