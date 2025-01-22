import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from monai.inferers import SlidingWindowInferer
from monai.utils import set_determinism
from light_training.evaluation.metric import dice
from light_training.trainer import Trainer
from light_training.utils.lr_scheduler import LinearWarmupCosineAnnealingLR
from light_training.utils.files_helper import save_new_model_and_delete_last
from monai.losses import DiceLoss
from unet.basic_unet_denose import BasicUNetDe
from unet.basic_unet import BasicUNetEncoder
from guided_diffusion.gaussian_diffusion import (
    get_named_beta_schedule,
    ModelMeanType,
    ModelVarType,
    LossType,
)
from guided_diffusion.respace import SpacedDiffusion, space_timesteps
from guided_diffusion.resample import UniformSampler

# Ensure reproducibility
set_determinism(123)

# Default dataset paths
default_data_path = r'F:\Re-DiffiNet'
default_log_path = os.path.join(default_data_path, 'log')

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Train a diffusion-based UNet model for medical image segmentation.")
parser.add_argument(
    "--data_path",
    type=str,
    default=os.path.join(default_data_path, "automatic"),
    help="Path to the dataset (automatic or expert)."
)
parser.add_argument(
    "--log_path",
    type=str,
    default=default_log_path,
    help="Path to save logs and model checkpoints."
)
parser.add_argument(
    "--fold_number",
    type=int,
    default=0,
    help="Fold number for cross-validation (default: 0)."
)
parser.add_argument(
    "--num_gpus",
    type=int,
    default=0,
    help="Number of GPUs to use (default: 0 for CPU)."
)
args = parser.parse_args()

# Validate dataset path
if not os.path.exists(args.data_path):
    raise ValueError(f"The specified data path does not exist: {args.data_path}")

# Configure device
device = torch.device("cuda" if torch.cuda.is_available() and args.num_gpus > 0 else "cpu")

# Define directories
logdir = os.path.join(args.log_path, f'fold{args.fold_number}')
os.makedirs(logdir, exist_ok=True)

# Hyperparameters and configurations
max_epoch = 200
batch_size = 2
val_every = 5
use_UNetopt = True
number_modality = 7 if use_UNetopt else 4
number_targets = 3  # WT, TC, ET

# Define the DiffUNet model
class DiffUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed_model = BasicUNetEncoder(
            in_channels=3,
            out_channels=number_modality,
            n_class=number_targets,
            features=[64, 64, 128, 256, 512, 64]
        )

        self.model = BasicUNetDe(
            in_channels=3,
            out_channels=number_modality + number_targets,
            n_class=number_targets,
            features=[64, 64, 128, 256, 512, 64],
            act=("LeakyReLU", {"negative_slope": 0.1, "inplace": False}),
        )

        betas = get_named_beta_schedule("linear", 1000)
        self.diffusion = SpacedDiffusion(
            use_timesteps=space_timesteps(1000, [1000]),
            betas=betas,
            model_mean_type=ModelMeanType.START_X,
            model_var_type=ModelVarType.FIXED_LARGE,
            loss_type=LossType.MSE,
        )

        self.sample_diffusion = SpacedDiffusion(
            use_timesteps=space_timesteps(1000, [50]),
            betas=betas,
            model_mean_type=ModelMeanType.START_X,
            model_var_type=ModelVarType.FIXED_LARGE,
            loss_type=LossType.MSE,
        )
        self.sampler = UniformSampler(1000)

    def forward(self, image=None, x=None, pred_type=None, step=None):
        if pred_type == "q_sample":
            noise = torch.randn_like(x).to(x.device)
            t, weight = self.sampler.sample(x.shape[0], x.device)
            return self.diffusion.q_sample(x, t, noise=noise), t, noise

        elif pred_type == "denoise":
            embeddings = self.embed_model(image)
            return self.model(x, t=step, image=image, embeddings=embeddings)

        elif pred_type == "ddim_sample":
            embeddings = self.embed_model(image)
            sample_out = self.sample_diffusion.ddim_sample_loop(
                self.model, (1, number_targets, 128, 192, 128),
                model_kwargs={"image": image, "embeddings": embeddings}
            )
            return sample_out["pred_xstart"]

# Trainer class to handle training logic
class BraTSTrainer(Trainer):
    def __init__(self, env_type, max_epochs, batch_size, device="cpu", val_every=1, num_gpus=1, logdir="./logs/"):
        super().__init__(env_type, max_epochs, batch_size, device, val_every, num_gpus, logdir)
        self.window_infer = SlidingWindowInferer(roi_size=[128, 192, 128], sw_batch_size=1, overlap=0.25)
        self.model = DiffUNet().to(device)

        self.best_mean_dice = 0.0
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-4, weight_decay=1e-3)
        self.scheduler = LinearWarmupCosineAnnealingLR(self.optimizer, warmup_epochs=30, max_epochs=max_epochs)
        self.bce = nn.BCEWithLogitsLoss()
        self.dice_loss = DiceLoss(sigmoid=True)

    def training_step(self, batch):
        image, label = self.get_input(batch)
        x_start = (label) * 2 - 1
        x_t, t, noise = self.model(x=x_start, pred_type="q_sample")
        pred_xstart = self.model(x=x_t, step=t, image=image, pred_type="denoise")

        loss_dice = self.dice_loss(pred_xstart, label)
        loss_bce = self.bce(pred_xstart, label)
        loss = loss_dice + loss_bce

        self.log("train_loss", loss, step=self.global_step)
        return loss

    def get_input(self, batch):
        image, label = batch["image"].to(self.device), batch["label"].to(self.device)
        return image, label

# Initialize and run the trainer
if __name__ == "__main__":
    trainer = BraTSTrainer(
        env_type="DDP" if args.num_gpus > 1 else "pytorch",
        max_epochs=max_epoch,
        batch_size=batch_size,
        device=device,
        val_every=val_every,
        num_gpus=args.num_gpus,
        logdir=logdir
    )
    if not hasattr(trainer, 'train'):
        raise ValueError("No training dataset provided!")
    trainer.train(train_dataset=None, val_dataset=None)  # Replace `None` with actual datasets.
