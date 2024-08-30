import os
import pandas as pd
from datasets.cimat import CimatDataset
from torch.utils.data import DataLoader


def prepare_dataloaders(base_dir, dataset, trainset, feat_channels):
    train_dataset = CimatDataset(
        base_dir=base_dir,
        dataset=dataset,
        trainset=trainset,
        features_channels=feat_channels,
        features_extension=".tiff",
        labels_extension=".pgm",
        learning_dir="trainingFiles",
        mode="train",
    )

    valid_dataset = CimatDataset(
        base_dir=base_dir,
        dataset=dataset,
        trainset=trainset,
        features_channels=feat_channels,
        features_extension=".tiff",
        labels_extension=".pgm",
        learning_dir="crossFiles",
        mode="cross",
    )

    test_dataset = CimatDataset(
        base_dir=base_dir,
        dataset=dataset,
        trainset=trainset,
        features_channels=feat_channels,
        features_extension=".tiff",
        labels_extension=".pgm",
        learning_dir="testingFiles",
        mode="test",
    )
    print(f"Training dataset length: {len(train_dataset)}")
    print(f"Validation dataset length: {len(valid_dataset)}")
    print(f"Testing dataset length: {len(test_dataset)}")

    train_dataloader = DataLoader(
        train_dataset, batch_size=16, pin_memory=True, shuffle=True, num_workers=12
    )
    valid_dataloader = DataLoader(
        valid_dataset, batch_size=4, pin_memory=True, shuffle=False, num_workers=4
    )
    test_dataloader = DataLoader(
        test_dataset, batch_size=4, pin_memory=True, shuffle=False, num_workers=4
    )
    return train_dataloader, valid_dataloader, test_dataloader
