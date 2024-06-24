# the code is based on the following repository: https://github.com/Vincent-ZHQ/MRDF/blob/main/model/avdf_multilabel.py
import argparse
import csv
import json
import os
import random

import numpy as np
import torch
from pytorch_lightning import Callback, LightningModule, Trainer, seed_everything

from models import AVDF, AVOC, MRDF_CE, MSOC, AVDF_Multilabel, Dissonance, MRDF_Margin
from new_datasets.dataloader import FakeavcelebDataModule as TalkNetDataModule

random_seed = 42

batch_size = 16
num_workers = 4
max_sample_size = 30
dataset_type = "new"

parser = argparse.ArgumentParser(description="MRDF training")
parser.add_argument("--model_type", type=str, default="MRDF_CE")
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--num_workers", type=int, default=1)

parser.add_argument("--margin_audio", type=float, default=0.0)
parser.add_argument("--margin_visual", type=float, default=0.0)
parser.add_argument("--margin_contrast", type=float, default=0.0)
parser.add_argument("--outputs", type=str, default="/data/kyungbok/outputs")

parser.add_argument("--loss_type", type=str, default="margin")
parser.add_argument("--wandb", action="store_true")
parser.add_argument("--max_frames", type=int, default=30)
parser.add_argument("--dataset_type", type=str, default="new")

parser.add_argument("--sync", action="store_true")
parser.add_argument("--use_threshold", action="store_true")
parser.add_argument("--audio_threshold", type=float, default=0.5)
parser.add_argument("--visual_threshold", type=float, default=0.5)
parser.add_argument("--final_threshold", type=float, default=0.5)

parser.add_argument("--ckpt", type=str, default="")

parser.add_argument("--name", type=str, default="")
parser.add_argument("--pred_strategy", type=str, default="mean")
parser.add_argument("--test_subset", type=str, default="all")
parser.add_argument("--file_name", type=str, default="")

parser.add_argument("--scnet", action="store_true")
parser.add_argument("--crop_face", action="store_true")
parser.add_argument("--oc_option", type=str, default="both")
parser.add_argument("--random_seed", type=int, default=42)


parser.add_argument("--middle_infer", action="store_true")
parser.add_argument("--save_score", action="store_true")
parser.add_argument("--save_features", action="store_true")
parser.add_argument("--score_fusion", action="store_true")


def log_to_file(message, file_path="training_log.txt"):
    with open(file_path, "a") as f:
        f.write(message + "\n")  # Append the message to the file along with a newline.


def append_to_csv(results, file_path="evaluation_results.csv"):
    # Check if the file exists to decide on writing headers or not
    file_exists = os.path.isfile(file_path)
    print(results)

    # Define your headers
    headers = [
        "model",
        "pred_method",
        "test_subset",
        "acc",
        "eer",
        "auc",
        "fake_f1score",
        "fake_precision",
        "fake_recall",
        "real_f1score",
        "real_precision",
        "real_recall",
        "seed",
    ]

    # Open the file in append mode ('a')
    with open(file_path, "a", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=headers)

        # Write the headers if the file is new
        if not file_exists:
            writer.writeheader()

        # Write the row of results
        writer.writerow(results)


def set_seed(seed):
    # seed init.
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    # torch seed init.
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False  # train speed is slower after enabling this opts.

    # https://pytorch.org/docs/stable/generated/torch.use_deterministic_algorithms.html
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"

    # avoiding nondeterministic algorithms (see https://pytorch.org/docs/stable/notes/randomness.html)
    torch.use_deterministic_algorithms(True)


def eval(args):

    # get name of the model's ckpt
    specific_name = args.model_type
    if args.oc_option != "both":
        specific_name += f"_{args.oc_option}"
    if args.scnet:
        specific_name += "_scnet"

    ckpt = f"/{args.outputs}/ckpts/{args.model_type}/new/{specific_name}_0.0002_seed:{args.random_seed}.ckpt"

    if args.score_fusion:
        specific_name += "_score_fusion"
    # check if the ckpt exists
    if not os.path.exists(ckpt):
        print(f"{ckpt} does not exist")
    else:
        print(ckpt)

    model_dict = {
        "MRDF_Margin": MRDF_Margin,
        "MRDF_CE": MRDF_CE,
        "AVDF": AVDF,
        "AVDF_Multilabel": AVDF_Multilabel,
    }

    stack_audio = True

    if "OC" in args.model_type or "Dissonance" in args.model_type:
        stack_audio = False

    args.stack_audio = stack_audio

    if args.model_type == "AVOC":
        model = AVOC(
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            positional_emb_flag=True,
            oc_option=args.oc_option,
            scnet=args.scnet,
        )
    elif args.model_type == "MSOC":
        model = MSOC(pred_strategy=args.pred_strategy, scnet=args.scnet)
    elif args.model_type == "Dissonance":
        model = Dissonance(learning_rate=args.learning_rate, weight_decay=args.weight_decay)

    else:
        model = model_dict[args.model_type](
            margin_contrast=args.margin_contrast,
            margin_audio=args.margin_audio,
            margin_visual=args.margin_visual,
            weight_decay=args.weight_decay,
            learning_rate=args.learning_rate,
            batch_size=args.batch_size,
        )

    model.eval()

    args.stack_audio = stack_audio

    dm = TalkNetDataModule(
        batch_size=batch_size,
        num_workers=num_workers,
        max_sample_size=max_sample_size,
        dataset_type=args.dataset_type,
        test_subset=args.test_subset,
        mask_face=args.crop_face,
        stack_audio=args.stack_audio,
    )

    trainer = Trainer(
        devices=[0],
        accelerator="auto",
    )

    result = trainer.test(model, datamodule=dm)[0]
    if "test_eer" not in result:
        result["test_eer"] = "N/A"
    csv_result = {
        "model": specific_name,
        "pred_method": args.pred_strategy,
        "test_subset": args.test_subset,
        "acc": result["test_acc"],
        "eer": result["test_eer"],
        "auc": result["test_auroc"],
        "fake_f1score": result["test_fake_f1score"],
        "fake_precision": result["test_fake_precision"],
        "fake_recall": result["test_fake_recall"],
        "real_f1score": result["test_real_f1score"],
        "real_precision": result["test_real_precision"],
        "real_recall": result["test_real_recall"],
        "seed": args.random_seed,
    }
    append_to_csv(csv_result, file_path=f"evaluation_results{args.file_name}.csv")


if __name__ == "__main__":
    set_seed(random_seed)
    args = parser.parse_args()
    eval(args)
