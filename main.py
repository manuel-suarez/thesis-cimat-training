import os
import time
import torch
import argparse
import pandas as pd
from torch.cuda import is_available

from data import CimatDataset
from dataloaders import prepare_dataloaders
from tqdm import tqdm
from torch import nn, optim
from torch.utils.data import DataLoader
from models.unet_resnet34 import UnetResNet34


def test_step(model, loss_fn, test_loader, device):
    TP = 0
    FP = 0
    TN = 0
    FN = 0
    total_loss = 0
    y_true = []
    y_pred = []
    model.eval()
    with torch.no_grad():
        for idx, (images, labels) in enumerate(test_loader):
            images, labels = images.to(device), labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = loss_fn(outputs, labels)
            total_loss += loss.item()

            # Eval
            y_pred.extend(outputs.cpu().detach().numpy())
            y_true.extend(labels.cpu().detach().numpy())

    # Conver lists to tensors for calculation
    y_true_tensor = torch.tensor(y_true)
    y_pred_tensor = torch.tensor(y_pred)

    # Metrics
    TP = ((y_pred_tensor == 1) & (y_true_tensor == 1)).sum().item()
    FP = ((y_pred_tensor == 1) & (y_true_tensor == 0)).sum().item()
    TN = ((y_pred_tensor == 0) & (y_true_tensor == 0)).sum().item()
    FN = ((y_pred_tensor == 0) & (y_true_tensor == 1)).sum().item()

    accuracy = (TP + TN) / (TP + FP + TN + FN)
    precision = TP / (TP + FP) if TP + FP > 0 else 0
    recall = TP / (TP + FN) if TP + FN > 0 else 0
    f1score = (
        2 * (precision * recall) / (precision + recall)
        if (precision + recall) > 0
        else 0
    )
    return {
        "loss": total_loss,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1score": f1score,
    }


def valid_step(model, loss_fn, valid_loader, device):
    TP = 0
    FP = 0
    TN = 0
    FN = 0
    total_loss = 0
    y_true = []
    y_pred = []
    model.eval()
    with torch.no_grad():
        for idx, (images, labels) in enumerate(valid_loader):
            images, labels = images.to(device), labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = loss_fn(outputs, labels)
            total_loss += loss.item()

            # Eval
            y_pred.extend(outputs.cpu().detach().numpy())
            y_true.extend(labels.cpu().detach().numpy())

    # Conver lists to tensors for calculation
    y_true_tensor = torch.tensor(y_true)
    y_pred_tensor = torch.tensor(y_pred)

    # Metrics
    TP = ((y_pred_tensor == 1) & (y_true_tensor == 1)).sum().item()
    FP = ((y_pred_tensor == 1) & (y_true_tensor == 0)).sum().item()
    TN = ((y_pred_tensor == 0) & (y_true_tensor == 0)).sum().item()
    FN = ((y_pred_tensor == 0) & (y_true_tensor == 1)).sum().item()

    accuracy = (TP + TN) / (TP + FP + TN + FN)
    precision = TP / (TP + FP) if TP + FP > 0 else 0
    recall = TP / (TP + FN) if TP + FN > 0 else 0
    f1score = (
        2 * (precision * recall) / (precision + recall)
        if (precision + recall) > 0
        else 0
    )
    return {
        "loss": total_loss,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1score": f1score,
    }


def train_step(model, loss_fn, train_loader, device):
    TP = 0
    FP = 0
    TN = 0
    FN = 0
    total_loss = 0
    y_true = []
    y_pred = []
    model.train(True)
    for idx, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)

        # Forward pass
        outputs = model(images)
        loss = loss_fn(outputs, labels)
        total_loss += loss.item()

        # Eval
        y_pred.extend(outputs.cpu().detach().numpy())
        y_true.extend(labels.cpu().detach().numpy())

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Conver lists to tensors for calculation
    y_true_tensor = torch.tensor(y_true)
    y_pred_tensor = torch.tensor(y_pred)

    # Metrics
    TP = ((y_pred_tensor == 1) & (y_true_tensor == 1)).sum().item()
    FP = ((y_pred_tensor == 1) & (y_true_tensor == 0)).sum().item()
    TN = ((y_pred_tensor == 0) & (y_true_tensor == 0)).sum().item()
    FN = ((y_pred_tensor == 0) & (y_true_tensor == 1)).sum().item()

    accuracy = (TP + TN) / (TP + FP + TN + FN)
    precision = TP / (TP + FP) if TP + FP > 0 else 0
    recall = TP / (TP + FN) if TP + FN > 0 else 0
    f1score = (
        2 * (precision * recall) / (precision + recall)
        if (precision + recall) > 0
        else 0
    )
    return {
        "loss": total_loss,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1score": f1score,
    }


def train_epoch(epoch, epochs, model, loss_fn, train_loader, valid_loader, device):
    train_metrics = train_step(model, loss_fn, train_loader, device)
    valid_metrics = valid_step(model, loss_fn, valid_loader, device)
    print(f"Epoch: {epoch+1}/{epochs}")
    print(
        f"\tTrain metrics: loss: {train_metrics['loss']}, accuracy: {train_metrics['accuracy']}, precision: {train_metrics['precision']}, recall: {train_metrics['recall']}, F1-score: {train_metrics['f1score']}"
    )
    print(
        f"\tValid metrics: loss: {valid_metrics['loss']}, accuracy: {valid_metrics['accuracy']}, precision: {valid_metrics['precision']}, recall: {valid_metrics['recall']}, F1-score: {valid_metrics['f1score']}"
    )


if __name__ == "__main__":
    # Load command arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset")
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

    trainset = "{:02}".format(slurm_array_task_id)
    print(f"Trainset: {trainset}")

    # Check if results path exists
    if not os.path.exists(args.results_path):
        os.makedirs(args.results_path, exist_ok=True)
    # Configure directories
    home_dir = os.path.expanduser("~")
    # Features to use
    feat_channels = ["ORIGIN", "ORIGIN", "VAR"]
    # Dataloaders
    train_loader, valid_loader, test_loader = prepare_dataloaders(
        base_dir=home_dir,
        dataset=args.dataset,
        trainset=trainset,
        feat_channels=feat_channels,
    )

    # Check device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device: ", device)

    # Load and configure model (segmentation_models_pytorch)
    model = UnetResNet34().to(device)
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    # logger = CSVLogger(os.path.join(f"logs_dataset{train_file}"), name="results.csv")

    # Training and Validation
    print("[INFO] training the network...")
    startTime = time.time()
    for epoch in range(args.num_epochs):
        train_epoch(
            epoch, args.num_epochs, model, loss_fn, train_loader, valid_loader, device
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

    # Testing
    test_metrics = test_step(model, loss_fn, test_loader, device)
    print(
        f"\tTest metrics: loss: {test_metrics['loss']}, accuracy: {test_metrics['accuracy']}, precision: {test_metrics['precision']}, recall: {test_metrics['recall']}, F1-score: {test_metrics['f1score']}"
    )
    # display total time
    endTime = time.time()
    print(
        "[INFO] total time taken to test the model: {:.2f}s".format(endTime - startTime)
    )
