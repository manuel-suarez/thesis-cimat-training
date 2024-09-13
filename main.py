import os
import time
import torch
import argparse
import lightning as L

from datasets import get_dataloaders
from models import get_model

from torch import nn, optim
from module import CimatModule
from lightning.pytorch.loggers import CSVLogger

# Currently we are implemented the loss and optimizer selection in main module, however in the future
# maybe we can move to his own modules depending on other selections or features


# Loss function depends on type of segmentation (depending on dataset name)
def get_loss(ds_name):
    # Most of problems are binary segmentation so we only need to distinguis on krestenitis dataset
    if ds_name == "krestenitis":
        return nn.CrossEntropyLoss()
    else:
        return nn.BCEWithLogitsLoss()


def get_trainer_configuration():
    # Loss function and optimizer
    loss_fn = get_loss(args.dataset)
    optimizer = get_optimizer()
    logger = CSVLogger("logs", name=f"cimat_dataset{args.dataset}_trainset{trainset}")
    module = CimatModule(model, optimizer, loss_fn)
    trainer = L.Trainer(
        max_epochs=int(args.num_epochs), devices=1, accelerator="gpu", logger=logger
    )
    return module, trainer


# Actually we are using Adam for all cases
def get_optimizer():
    return optim.Adam(model.parameters(), lr=1e-4)


def training_step(trainer, module, dataloaders):
    # Extract dataloaders
    train_dataloader, valid_dataloader, _ = dataloaders
    print("[INFO] training the network...")
    startTime = time.time()
    trainer.fit(
        model=module,
        train_dataloaders=train_dataloader,
        val_dataloaders=valid_dataloader,
    )
    # display total time
    endTime = time.time()
    trainer.save_checkpoint(
        f"cimat_dataset{args.dataset}_trainset{trainset}-best_model.ckpt"
    )
    print(
        "[INFO] total time taken to train the model: {:.2f}s".format(
            endTime - startTime
        )
    )


def testing_step(trainer, module, dataloaders):
    _, _, test_dataloader = dataloaders
    print("[INFO] testing the network...")
    startTime = time.time()
    trainer.test(model=module, dataloaders=test_dataloader)
    # display total time
    endTime = time.time()
    print(
        "[INFO] total time taken to test the model: {:.2f}s".format(endTime - startTime)
    )


if __name__ == "__main__":
    # Load command arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset")
    parser.add_argument("model_arch")
    parser.add_argument("--model_encoder", required=False)

    # parser.add_argument("results_path")
    # parser.add_argument("num_epochs")
    args = parser.parse_args()
    print(args)

    # Use SLURM array environment variables to determine training and cross validation set number
    # slurm_array_job_id = int(os.getenv("SLURM_ARRAY_JOB_ID"))
    # slurm_array_task_id = int(os.getenv("SLURM_ARRAY_TASK_ID"))
    # slurm_node_list = os.getenv("SLURM_JOB_NODELIST")
    # print(f"SLURM_ARRAY_JOB_ID: {slurm_array_job_id}")
    # print(f"SLURM_ARRAY_TASK_ID: {slurm_array_task_id}")
    # print(f"SLURM_JOB_NODELIST: {slurm_node_list}")

    # trainset = "{:02}".format(slurm_array_task_id)
    # print(f"Trainset: {trainset}")

    # Check if results path exists
    # if not os.path.exists(args.results_path):
    #    os.makedirs(args.results_path, exist_ok=True)
    # Configure directories
    home_dir = os.path.expanduser("~")
    # Features to use
    # feat_channels = ["ORIGIN", "ORIGIN", "VAR"]

    # Dataloaders
    dataloaders = get_dataloaders(home_dir, args.dataset, args)
    # Model
    model = get_model(args.model_arch, args.model_encoder)
    # Training configuration
    module, trainer = get_trainer_configuration()
    # Training step
    training_step(trainer, module, dataloaders)
    # Testing step
    testing_step(trainer, module, dataloaders)

    from matplotlib import pyplot as plt

    # Test example segmentations
    results_dir = os.path.join(
        "results", f"cimat_dataset{args.dataset}_trainset{trainset}", "figures"
    )
    os.makedirs(results_dir, exist_ok=True)
    checkpoint = torch.load(
        f"cimat_dataset{args.dataset}_trainset{trainset}-best_model.ckpt"
    )
    print(checkpoint.keys())
    model.eval()
    for directory, loader in zip(
        ["train", "valid", "test"],
        [train_dataloader, valid_dataloader, test_dataloader],
    ):
        figures_dir = os.path.join(results_dir, directory)
        os.makedirs(figures_dir, exist_ok=True)
        for images, labels in loader:
            predictions = model(images)
            print("Images shape: ", images.shape)
            print("Labels shape: ", labels.shape)
            print("Preds shape: ", predictions.shape)

            images, labels, predictions = (
                images.detach().numpy(),
                labels.detach().numpy(),
                predictions.detach().numpy(),
            )

            fig, axs = plt.subplots(1, 3, figsize=(12, 8))
            axs[0].imshow(images[0, 0, :, :])
            axs[1].imshow(predictions[0, 0, :, :])
            axs[2].imshow(labels[0, 0, :, :])
            plt.savefig(os.path.join(figures_dir, "result.png"))
            plt.close()
