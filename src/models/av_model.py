import fairseq
import hubert
import hubert_asr
import hubert_pretraining
import torch
import torch.nn as nn

from models.mc_tnc import MultiscaleMultibranchTCN
from util.constants import (
    HUBERT_ENCODER_MODEL_STATE_DICT_PATH,
    MC_TCN_MODEL_PATH,
    MODEL_WEIGHTS_PATH,
)


class AVModel(nn.Module):
    def __init__(self, pretrained: bool, device: str):
        super(AVModel, self).__init__()
        self.device = device
        self._init_hubert_encoder(pretrained=pretrained)
        self._init_tsm_model(pretrained=pretrained)

    def _init_hubert_encoder(self, pretrained: bool = True):
        def load_hubert_encoder(pretrained: bool):
            # load hubert encoder
            print(f"Creating hubert encoder from checkpoint")
            ckpt_path = "finetune-model.pt"
            models, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([ckpt_path])
            model = models[0]
            if hasattr(models[0], "decoder"):
                print(f"Checkpoint: fine-tuned")
                model = models[0].encoder.w2v_model
            else:
                print(f"Checkpoint: pre-trained w/o fine-tuning")

            # load pretrained weights
            if pretrained:
                model.load_state_dict(torch.load(HUBERT_ENCODER_MODEL_STATE_DICT_PATH))

            return model

        self.hubert_encoder = load_hubert_encoder(pretrained=pretrained)
        self.hubert_encoder.to(self.device)

    def _init_tsm_model(self, pretrained: bool = True):
        if pretrained:
            self.ms_tcn = torch.load(MC_TCN_MODEL_PATH)
        else:
            hidden_dim = 256
            num_classes = 2
            input_size = 768
            relu_type = "swish"

            tcn_options = {
                "num_layers": 4,
                "kernel_size": [3, 5, 7],
                "dropout": 0.2,
                "dwpw": False,
                "width_mult": 1,
            }
            self.ms_tcn = MultiscaleMultibranchTCN(
                input_size=input_size,
                num_channels=[hidden_dim * len(tcn_options["kernel_size"]) * tcn_options["width_mult"]]
                * tcn_options["num_layers"],
                num_classes=num_classes,
                tcn_options=tcn_options,
                dropout=tcn_options["dropout"],
                relu_type=relu_type,
                dwpw=tcn_options["dwpw"],
            )
            self.ms_tcn.to(self.device)

    def forward(self, lip_frames, stacked_spectrogram):
        feature, _ = self.hubert_encoder.extract_finetune(
            source={"video": lip_frames, "audio": stacked_spectrogram},
            padding_mask=None,
            output_layer=None,
        )  # [B, T, F]
        print(f"Hubert Feature shape: {feature.shape}")
        B, T, F = feature.shape
        length = [T] * B
        feature, output = self.ms_tcn(feature, length, B)
        return output, feature
