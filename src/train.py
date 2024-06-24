import argparse
import os
import random
import re

import numpy as np
import torch
from pytorch_lightning import Callback, LightningModule, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import WandbLogger

import wandb
from models import AVDF, AVOC, MRDF_CE, MSOC, AVDF_Multilabel, Dissonance, MRDF_Margin
from new_datasets.dataloader import FakeavcelebDataModule


class EarlyStoppingLR(Callback):
    """Early stop model training when the LR is lower than threshold."""

    def __init__(self, lr_threshold: float, mode="all"):
        self.lr_threshold = lr_threshold

        if mode in ("any", "all"):
            self.mode = mode
        else:
            raise ValueError(f"mode must be one of ('any', 'all')")

    def on_train_epoch_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self._run_early_stop_checking(trainer)

    def _run_early_stop_checking(self, trainer: Trainer) -> None:
        metrics = trainer._logger_connector.callback_metrics
        if len(metrics) == 0:
            return
        all_lr = []
        for key, value in metrics.items():
            if re.match(r"opt\d+_lr\d+", key):
                all_lr.append(value)

        if len(all_lr) == 0:
            return

        if self.mode == "all":
            if all(lr <= self.lr_threshold for lr in all_lr):
                trainer.should_stop = True
        elif self.mode == "any":
            if any(lr <= self.lr_threshold for lr in all_lr):
                trainer.should_stop = True


parser = argparse.ArgumentParser(description="MRDF training")
parser.add_argument("--dataset", type=str, default="fakeavceleb")
parser.add_argument("--model_type", type=str, default="MRDF_CE")
parser.add_argument("--data_root", type=str, default="/data/kyungbok/FakeAVCeleb_v1.2/")
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--num_workers", type=int, default=4)

parser.add_argument("--gpu", type=int, default=2)

parser.add_argument("--precision", default=16)
parser.add_argument("--num_train", type=int, default=None)
parser.add_argument("--num_val", type=int, default=None)
parser.add_argument("--max_epochs", type=int, default=30)
parser.add_argument("--min_epochs", type=int, default=30)
parser.add_argument("--epochs", type=int, default=30)
parser.add_argument("--patience", type=int, default=0)
parser.add_argument("--log_steps", type=int, default=20)
parser.add_argument("--resume", type=str, default=None)

parser.add_argument("--learning_rate", type=float, default=2e-4)
parser.add_argument("--weight_decay", type=float, default=1e-4)
parser.add_argument("--margin_audio", type=float, default=0.0)
parser.add_argument("--margin_visual", type=float, default=0.0)
parser.add_argument("--margin_contrast", type=float, default=0.0)
parser.add_argument("--outputs", type=str, default="/data/kyungbok/outputs")
parser.add_argument("--loss_type", type=str, default="margin")
parser.add_argument("--wandb", action="store_true")
parser.add_argument("--max_frames", type=int, default=30)
parser.add_argument("--dataset_type", type=str, default="new")
parser.add_argument("--pred_strategy", type=str, default="mean")

parser.add_argument("--projection", action="store_true")

parser.add_argument("--project_name", type=str, default="final_test")


parser.add_argument("--name", type=str, default="")

parser.add_argument("--light", action="store_true")
parser.add_argument("--random_seed", type=int, default=42)

parser.add_argument("--oc_option", type=str, default="both")

parser.add_argument("--file_name", type=str, default="")
# parser.add_argument("--resnet3d", action="store_true")
parser.add_argument("--scnet", action="store_true")


def dict_to_str(src_dict):
    dst_str = ""
    for key in src_dict.keys():
        dst_str += " %s: %.4f " % (key, src_dict[key])
    return dst_str


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


if __name__ == "__main__":
    args = parser.parse_args()

    learning_rate = args.learning_rate
    weight_decay = args.weight_decay
    dataset = args.dataset

    print("pytorch version: ", torch.__version__)
    print("cuda version: ", torch.version.cuda)
    print("cudnn version: ", torch.backends.cudnn.version())
    print("gpu name: ", torch.cuda.get_device_name())
    print("gpu index: ", torch.cuda.current_device())

    results = []
    args.file_name = args.learning_rate
    # wandb.init(project="margin", name=args.log_name)
    model_dict = {
        "MRDF_Margin": MRDF_Margin,
        "MRDF_CE": MRDF_CE,
        "AVDF": AVDF,
        "AVDF_Multilabel": AVDF_Multilabel,
    }

    # for one_run in [42]:
    set_seed(args.random_seed)
    name = f"{args.name}_{args.learning_rate}"
    wandb_name = f"{args.name}_{args.random_seed}"

    if args.wandb:
        wandb.finish()
        # Start a new WandB run with a unique name for the current fold

        wandb.init(
            project=args.project_name,
            reinit=True,
            name=wandb_name,
            config={
                "model_type": args.name,
                "batch_size": args.batch_size,
                "learning_rate": args.learning_rate,
                "weight_decay": args.weight_decay,
            },
        )

        # Create a new WandbLogger for PyTorch Lightning with the current WandB run
        # Note: No need to pass log_model="all" here, as it's a parameter of `wandb.init`, not `WandbLogger`
        wandb_logger = WandbLogger(experiment=wandb.run)
    args.save_name_id = name + "_seed:" + str(args.random_seed)
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
            weight_decay=weight_decay,
            learning_rate=learning_rate,
            batch_size=args.batch_size,
        )

    dm = FakeavcelebDataModule(
        root=args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_sample_size=args.max_frames,
        take_train=args.num_train,
        take_dev=args.num_val,
        dataset_type=args.dataset_type,
        stack_audio=args.stack_audio,
    )

    try:
        precision = int(args.precision)
    except ValueError:
        precision = args.precision

    monitor = "val_auroc"
    early_stop_callback = EarlyStopping(
        monitor=monitor, min_delta=0.00, patience=args.patience, verbose=False, mode="max"
    )

    trainer = Trainer(
        log_every_n_steps=args.log_steps,
        precision=precision,
        min_epochs=args.epochs,
        max_epochs=args.epochs,
        callbacks=[
            ModelCheckpoint(
                dirpath=f"{args.outputs}/ckpts/{args.model_type}/{args.dataset_type}",
                save_last=False,
                filename=args.save_name_id,
                monitor=monitor,
                mode="max",
            ),
            EarlyStoppingLR(lr_threshold=1e-7),
            early_stop_callback,
        ],
        enable_checkpointing=True,
        benchmark=True,
        num_sanity_val_steps=0,
        deterministic="warn",
        accelerator="auto",
        devices=[0],
        logger=wandb_logger,
    )

    trainer.fit(model, dm)

    # test
    model.eval()
    result = trainer.test(model, dm.test_dataloader(), ckpt_path="best")
    results.append(result)
    print(result)

    def dict_mean(dict_list):
        mean_dict = {}
        for key in dict_list[0][0].keys():
            mean_dict[key] = sum(d[0][key] for d in dict_list) / len(dict_list)
        return mean_dict

    print(results)
    wandb.finish()
