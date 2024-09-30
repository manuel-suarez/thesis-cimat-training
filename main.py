import os
import time
import torch
import argparse
import numpy as np
import lightning as L

from datasets import get_dataloaders, get_savers
from models import get_model

from PIL import Image
from torch import nn, optim
from module import CimatModule
from matplotlib import pyplot as plt
from lightning.pytorch.loggers import CSVLogger
from skimage.io import imsave


def get_model_args(ds_name):
    if ds_name == "krestenitis":
        return {"in_channels": 1, "out_channels": 5}
    if ds_name == "cimat":
        return {"in_channels": 3, "out_channels": 1, "wavelet_mode": True}
    if ds_name == "chn6_cug":
        return {"in_channels": 3, "out_channels": 1}
    if ds_name == "sos":
        return {"in_channels": 1, "out_channels": 1}


def get_problem_type(ds_name):
    if ds_name in ["cimat", "krestenitis", "sos"]:
        return "oil_spill"
    if ds_name in ["chn6_cug"]:
        return "road"
    if ds_name in ["chase_db1", "stare", "drive"]:
        return "retinal"
    raise Exception(
        f"No existe un tipo de problema registrado para el dataset {ds_name}"
    )


def build_dataset_name(ds_name, ds_args):
    # For Cimat dataset we need the aditional parameters (however these were tested before this function call so we don't need to test again)
    # For other datasets:
    #   We are using the dataset_num as a default value 01 because in the future maybe we can use data augmentation
    #   We are using the trainset_num as an indicator for different permutations
    dataset_num = ds_args["dataset_num"]
    trainset_num = ds_args["trainset_num"]
    if ds_name == "cimat":
        dataset_channels = ds_args["dataset_channels"]
        return f"{ds_name}_dataset-{dataset_num}_trainset-{trainset_num}_channels-{dataset_channels}"
    else:
        return f"{ds_name}_dataset-{dataset_num}_trainset-{trainset_num}"


def configure_results_path(
    results_path,
    ds_name,
    ds_args,
    model_arch,
    model_encoder,
    loss_fn,
    optimizer,
    epochs,
):
    # We need to create results directories according to the files that will be saved in them
    if not (("dataset_num" in ds_args) and ("trainset_num" in ds_args)):
        raise Exception(
            f"No se proporcionaron los argumentos necesarios para el dataset {ds_name}"
        )
    if (ds_name == "cimat") and not (
        ("dataset_num" in ds_args)
        and ("trainset_num" in ds_args)
        and ("dataset_channels" in ds_args)
    ):
        raise Exception(
            f"No se proporcionaron los argumentos necesarios para el dataset Cimat (dataset_num, trainset_num, dataset_channels)"
        )

    # The configuration must be:
    # result_base
    # - problem_type{oil_spill|road|retinal}
    #  - dataset_name{cimat|krestenitis|sos|chn6_cug|chase_db1|stare|drive}+dataset_num{17,19,20}+trainset_num{01-30}, currently only cimat dataset has dataset_num and trainset_num
    #   - architecture{unet,unetpp,fpn,linknet,pspnet,manet,deeplabv3p}+encoder[optional]{vgg16,vgg19,resnet18,resnet34,senet,efficientnetb0}
    #    - loss_fn+optimizer
    #     - epochs
    if model_encoder != None:
        model_name = model_arch + "_" + model_encoder
    else:
        model_name = model_arch
    results_dir = os.path.join(
        results_path,
        get_problem_type(ds_name),
        build_dataset_name(ds_name, ds_args),
        model_name,
        loss_fn + "_" + optimizer,
        f"epochs_{epochs}",
    )
    os.makedirs(results_dir, exist_ok=True)
    # Ahora generamos los directorios para los diferentes resultados
    os.makedirs(os.path.join(results_dir, "checkpoints"), exist_ok=True)
    os.makedirs(os.path.join(results_dir, "metrics"), exist_ok=True)
    os.makedirs(os.path.join(results_dir, "metrics", "lightning_logs"), exist_ok=True)
    os.makedirs(os.path.join(results_dir, "predictions"), exist_ok=True)
    os.makedirs(os.path.join(results_dir, "figures"), exist_ok=True)
    # Retornamos el directorio base para su uso en subsecuentes funciones
    return results_dir


# Currently we are implemented the loss and optimizer selection in main module, however in the future
# maybe we can move to his own modules depending on other selections or features


# Loss function depends on type of segmentation (depending on dataset name)
def get_loss(ds_name):
    # Most of problems are binary segmentation so we only need to distinguis on krestenitis dataset
    if ds_name == "krestenitis":
        return nn.CrossEntropyLoss()
    return nn.BCEWithLogitsLoss()


def get_loss_name(ds_name):
    if ds_name == "krestenitis":
        return "crossentropyloss"
    return "bcewithlogitsloss"


def get_trainer_configuration(ds_name, model, epochs, results_dir):
    # Loss function and optimizer
    loss_fn = get_loss(args.dataset)
    optimizer = get_optimizer(model)
    logger = CSVLogger(
        os.path.join(results_dir, "metrics"),
    )
    if ds_name == "krestenitis":
        module = CimatModule(model, optimizer, loss_fn, "multiclass", 5)
    else:
        module = CimatModule(model, optimizer, loss_fn)
    trainer = L.Trainer(
        max_epochs=int(epochs), devices=1, accelerator="gpu", logger=logger
    )
    return module, trainer


# Actually we are using Adam for all cases
def get_optimizer(model):
    return optim.Adam(model.parameters(), lr=1e-4)


def get_optimizer_name():
    return "adam"


def training_step(trainer, module, dataloaders, results_dir):
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
        os.path.join(results_dir, "checkpoints", "training_model.ckpt")
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


def predictions_step(
    model, dataloaders, results_dir, save_predictions, save_figures, max_batchs=10
):
    # Test example segmentations
    figures_dir = os.path.join(results_dir, "figures")
    predictions_dir = os.path.join(results_dir, "predictions")
    print("[INFO] generating predictions...")
    startTime = time.time()

    model.eval()
    train_dataloader, valid_dataloader, test_dataloader = dataloaders
    for directory, loader in zip(
        ["train", "valid", "test"],
        [train_dataloader, valid_dataloader, test_dataloader],
    ):
        # Generate predictions
        loader_predictions_dir = os.path.join(predictions_dir, directory)
        loader_figures_dir = os.path.join(figures_dir, directory)
        os.makedirs(loader_predictions_dir, exist_ok=True)
        os.makedirs(loader_figures_dir, exist_ok=True)
        # Generate figures and predictions
        for idx_batch, (images, labels) in enumerate(loader):
            predictions = model(images)

            images, labels, predictions = (
                images.detach().numpy(),
                labels.detach().numpy(),
                predictions.detach().numpy(),
            )
            # print("Shapes:")
            # print("\tImages: ", images.shape)
            # print("\tLabels: ", labels.shape)
            # print("\tPredictions: ", predictions.shape)

            # Save images, labels, predictions (batch size)
            save_predictions(
                idx_batch, images, labels, predictions, loader_predictions_dir
            )

            # Save figures
            save_figures(idx_batch, images, labels, predictions, loader_figures_dir)

            if idx_batch == max_batchs:
                break

    # display total time
    endTime = time.time()
    print(
        "[INFO] total time taken to generate predictions: {:.2f}s".format(
            endTime - startTime
        )
    )


if __name__ == "__main__":
    # Load command arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("results_path")
    parser.add_argument("dataset")
    parser.add_argument("architecture")
    parser.add_argument("epochs")
    parser.add_argument("--model_encoder", required=False)
    parser.add_argument("--model_channels", required=False)
    parser.add_argument("--dataset_num", required=False)
    parser.add_argument("--trainset_num", required=False)
    parser.add_argument("--dataset_channels", required=False)

    # parser.add_argument("results_path")
    # parser.add_argument("num_epochs")
    args = parser.parse_args()
    print(args)

    # Generic variables
    ds_name = args.dataset
    model_arch = args.architecture
    model_encoder = args.model_encoder
    epochs = args.epochs
    ds_args = {}
    if args.dataset_num:
        ds_args["dataset_num"] = args.dataset_num
    if args.trainset_num:
        ds_args["trainset_num"] = args.trainset_num
    if args.dataset_channels:
        ds_args["dataset_channels"] = args.dataset_channels

    # Use SLURM array environment variables to determine training and cross validation set number
    # If there is a command line argument we are using instead the environment variable (it takes precedence)
    slurm_array_job_id = os.getenv("SLURM_ARRAY_JOB_ID")
    slurm_array_task_id = os.getenv("SLURM_ARRAY_TASK_ID")
    slurm_node_list = os.getenv("SLURM_JOB_NODELIST")
    print(f"SLURM_ARRAY_JOB_ID: {slurm_array_job_id}")
    print(f"SLURM_ARRAY_TASK_ID: {slurm_array_task_id}")
    print(f"SLURM_JOB_NODELIST: {slurm_node_list}")

    if slurm_array_task_id is not None:
        ds_args["trainset_num"] = "{:02}".format(int(slurm_array_task_id))

    print("Dataset arguments:")
    print(ds_args)
    # Check if results path exists
    # if not os.path.exists(args.results_path):
    #    os.makedirs(args.results_path, exist_ok=True)
    # Configure directories
    home_dir = os.path.expanduser("~")
    # Features to use
    # feat_channels = ["ORIGIN", "ORIGIN", "VAR"]

    # Results path
    results_dir = configure_results_path(
        args.results_path,
        ds_name,
        ds_args,
        model_arch,
        model_encoder,
        get_loss_name(ds_name),
        get_optimizer_name(),
        epochs,
    )
    # Dataloaders
    dataloaders = get_dataloaders(home_dir, ds_name, ds_args)
    # Model
    model = get_model(model_arch, get_model_args(ds_name), model_encoder)
    if model == None:
        raise Exception("Error en la configuración del modelo")
    # Training configuration
    module, trainer = get_trainer_configuration(ds_name, model, epochs, results_dir)
    # Functions to save results
    save_predictions, save_figures = get_savers(ds_name)
    # Training step
    training_step(trainer, module, dataloaders, results_dir)
    # Testing step
    testing_step(trainer, module, dataloaders)
    # Predictions step
    predictions_step(model, dataloaders, results_dir, save_predictions, save_figures)

    print("Done!")
