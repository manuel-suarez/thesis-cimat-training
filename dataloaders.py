import os
import pandas as pd
from data import CimatDataset
from torch.utils.data import DataLoader

def prepare_dataloaders(base_dir, feat_channels, train_file, cross_file):
    data_dir = os.path.join(
        base_dir,
        "data",
        "projects",
        "consorcio-ia",
        "data",
        "oil_spills_17",
        "augmented_dataset",
    )
    feat_dir = os.path.join(data_dir, "features")
    labl_dir = os.path.join(data_dir, "labels")
    train_dir = os.path.join(data_dir, "learningCSV", "trainingFiles")
    cross_dir = os.path.join(data_dir, "learningCSV", "crossFiles")

    # Load CSV key files
    train_set = pd.read_csv(os.path.join(train_dir, f"train{train_file}.csv"))
    valid_set = pd.read_csv(os.path.join(cross_dir, f"cross{cross_file}.csv"))
    print(f"Training CSV file length: {len(train_set)}")
    print(f"Validation CSV file length: {len(valid_set)}")

    # Load generators
    train_keys = train_set["key"]
    valid_keys = valid_set["key"]
    train_dataset = CimatDataset(
        keys=train_keys,
        features_path=feat_dir,
        features_ext=".tiff",
        features_channels=feat_channels,
        labels_path=labl_dir,
        labels_ext=".pgm",
        dimensions=[224, 224, 3],
    )

    valid_dataset = CimatDataset(
        keys=valid_keys,
        features_path=feat_dir,
        features_ext=".tiff",
        features_channels=feat_channels,
        labels_path=labl_dir,
        labels_ext=".pgm",
        dimensions=[224, 224, 3],
    )
    print(f"Training dataset length: {len(train_dataset)}")
    print(f"Validation dataset length: {len(valid_dataset)}")

    train_dataloader = DataLoader(
        train_dataset, batch_size=16, pin_memory=True, shuffle=True, num_workers=12
    )
    valid_dataloader = DataLoader(
        valid_dataset, batch_size=4, pin_memory=True, shuffle=False, num_workers=4
    )
    return train_dataloader, valid_dataloader
