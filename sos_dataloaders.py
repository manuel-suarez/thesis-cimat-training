import os
import pandas as pd
from datasets.sos import SOSDataset
from torch.utils.data import DataLoader


def prepare_dataloaders(base_dir):
    data_dir = os.path.join(base_dir, "data", "SOS_dataset")

    train_dir = os.path.join(data_dir, "train")
    train_dataset = SOSDataset(
        base_dir=train_dir,
    )

    valid_dir = os.path.join(data_dir, "val")
    valid_dataset = SOSDataset(
        base_dir=valid_dir,
    )

    test_dir = os.path.join(data_dir, "test")
    test_dataset = SOSDataset(
        base_dir=test_dir,
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
