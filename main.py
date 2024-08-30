import os
import time
import torch
import argparse
import pandas as pd

from data import CimatDataset
from dataloaders import prepare_dataloaders
from tqdm import tqdm
from torch import nn, optim
from torch.utils.data import DataLoader
from models.unet_resnet34 import UnetResNet34

if __name__ == "__main__":
    # Load command arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("results_path")
    parser.add_argument("num_epochs")
    args = parser.parse_args()
    print(args)

    # Use SLURM array environment variables to determine training and cross validation set number
    slurm_array_job_id = int(os.getenv("SLURM_ARRAY_JOB_ID"))
    slurm_array_task_id = int(os.getenv("SLURM_ARRAY_TASK_ID"))
    slurm_node_list = os.getenv("SLURM_JOB_NODELIST")
    print(f"SLURM_ARRAY_JOB_ID: {slurm_array_job_id}")
    print(f"SLURM_ARRAY_TASK_ID: {slurm_array_task_id}")
    print(f"SLURM_JOB_NODELIST: {slurm_node_list}")

    train_file = "{:02}".format(slurm_array_task_id)
    cross_file = "{:02}".format(slurm_array_task_id)
    print(f"Train file: {train_file}")
    print(f"Cross file: {cross_file}")

    # Check if results path exists
    if not os.path.exists(args.results_path):
        os.makedirs(args.results_path, exist_ok=True)
    # Configure directories
    home_dir = os.path.expanduser("~")
    # Features to use
    feat_channels = ["ORIGIN", "ORIGIN", "VAR"]
    # Dataloaders
    train_dataloader, valid_dataloader = prepare_dataloaders(
        home_dir, feat_channels, train_file, cross_file
    )

    # Load and configure model (segmentation_models_pytorch)
    model = UnetResNet34()
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    logger = CSVLogger(os.path.join(f"logs_dataset{train_file}"), name="results.csv")
    module = CimatModule(model, optimizer, loss_fn)
    trainer = L.Trainer(
        max_epochs=int(args.num_epochs), devices=2, accelerator="gpu", logger=logger
    )
    # Training
    print("[INFO] training the network...")
    startTime = time.time()
    trainer.fit(
        model=module,
        train_dataloaders=train_dataloader,
        val_dataloaders=valid_dataloader,
    )
    # display total time
    endTime = time.time()
    print(
        "[INFO] total time taken to train the model: {:.2f}s".format(
            endTime - startTime
        )
    )
