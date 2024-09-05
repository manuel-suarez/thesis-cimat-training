import os
import time
import argparse
import lightning as L

from chn6_dataloaders import prepare_dataloaders
from torch import nn, optim
from models.unet_resnet34 import UnetResNet34
from module import CimatModule
from lightning.pytorch.loggers import CSVLogger

if __name__ == "__main__":
    # Load command arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("results_path")
    parser.add_argument("num_epochs")
    args = parser.parse_args()
    print(args)
    print("CHN6")

    # Use SLURM array environment variables to determine training and cross validation set number
    slurm_array_job_id = int(os.getenv("SLURM_ARRAY_JOB_ID"))
    slurm_array_task_id = int(os.getenv("SLURM_ARRAY_TASK_ID"))
    slurm_node_list = os.getenv("SLURM_JOB_NODELIST")
    print(f"SLURM_ARRAY_JOB_ID: {slurm_array_job_id}")
    print(f"SLURM_ARRAY_TASK_ID: {slurm_array_task_id}")
    print(f"SLURM_JOB_NODELIST: {slurm_node_list}")

    trainset = "{:02}".format(slurm_array_task_id)
    print(f"Trainset: {trainset}")

    # Check if results path exists
    if not os.path.exists(args.results_path):
        os.makedirs(args.results_path, exist_ok=True)
    # Configure directories
    home_dir = os.path.expanduser("~")
    # Dataloaders
    train_dataloader, valid_dataloader, test_dataloader = prepare_dataloaders(
        base_dir=home_dir,
    )

    # Load and configure model (segmentation_models_pytorch)
    model = UnetResNet34(in_channels=3)
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    logger = CSVLogger(os.path.join(f"logs_chn6_dataset"), name="results.csv")
    module = CimatModule(model, optimizer, loss_fn)
    trainer = L.Trainer(
        max_epochs=int(args.num_epochs), devices=1, accelerator="gpu", logger=logger
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
    print("[INFO] testing the network...")
    startTime = time.time()
    trainer.test(model=module, dataloaders=test_dataloader)
    # display total time
    endTime = time.time()
    print(
        "[INFO] total time taken to test the model: {:.2f}s".format(endTime - startTime)
    )
