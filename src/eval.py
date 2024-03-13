import argparse

from pytorch_lightning import Callback, LightningModule, Trainer, seed_everything

from models.mrdf_margin import MRDF_Margin
from models.msoc_all import MSOC
from new_datasets.fakeavceleb_light import FakeavcelebDataModule

train_fold = None
batch_size = 32
num_workers = 4
max_sample_size = 30
dataset_type = "new"

parser = argparse.ArgumentParser(description="MRDF training")
parser.add_argument("--model_type", type=str, default="MRDF_CE")
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--num_workers", type=int, default=16)

parser.add_argument("--margin_audio", type=float, default=0.0)
parser.add_argument("--margin_visual", type=float, default=0.0)
parser.add_argument("--margin_contrast", type=float, default=0.0)
parser.add_argument("--outputs", type=str, default="/data/kyungbok/outputs")

parser.add_argument("--loss_type", type=str, default="margin")
parser.add_argument("--wandb", action="store_true")
parser.add_argument("--max_frames", type=int, default=30)
parser.add_argument("--dataset_type", type=str, default="original")

parser.add_argument("--sync", action="store_true")
parser.add_argument("--use_threshold", action="store_true")
parser.add_argument("--audio_threshold", type=float, default=0.5)
parser.add_argument("--visual_threshold", type=float, default=0.5)
parser.add_argument("--final_threshold", type=float, default=0.5)

parser.add_argument("--ckpt", type=str, default="")

parser.add_argument("--name", type=str, default="")


def eval(args):
    ckpt = f"/data/kyungbok/outputs/ckpts/{args.ckpt}"
    if args.model_type == "MSOC":
        model = MSOC().load_from_checkpoint(ckpt, strict=False)
        model.sync = args.sync
        model.threshold = args.use_threshold
        model.audio_threshold = args.audio_threshold
        model.video_threshold = args.visual_threshold
        model.final_threshold = args.final_threshold
    elif args.model_type == "MRDF_Margin":
        model = MRDF_Margin().load_from_checkpoint(ckpt, strict=False)

    dm = FakeavcelebDataModule(
        train_fold=train_fold,
        batch_size=batch_size,
        num_workers=num_workers,
        max_sample_size=max_sample_size,
        dataset_type=dataset_type,
    )

    trainer = Trainer(devices=[0], accelerator="auto")

    trainer.test(model, datamodule=dm)


if __name__ == "__main__":
    args = parser.parse_args()
    eval(args)
