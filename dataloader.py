import os
from typing import Tuple
from torch.utils.data import DataLoader
import torch


def get_dataloader(dataset,
                   train_percentage: float = 0.8,
                   batch_size: int = 8,
                   save_model_and_dataloader: bool = False,
                   save_name: str = None
                   ) -> Tuple[DataLoader, DataLoader, DataLoader]:

    # split dataset train / val / test
    train_size = int(train_percentage * len(dataset))
    val_size = int(((1 - train_percentage) * 0.5) * len(dataset))
    test_size = len(dataset) - train_size - val_size
    print(f"\nDataset Size: {len(dataset)}\nTrain size: {train_size}\nValidation size: {val_size}\nTest size: {test_size}\n")

    train_dataset, temp_dataset = torch.utils.data.random_split(dataset, [train_size, val_size + test_size])
    val_dataset, test_dataset = torch.utils.data.random_split(temp_dataset, [val_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    assert len(train_loader) > 0, "empty train loader"

    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    assert len(val_loader) > 0, "empty validation loader"

    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    assert len(test_loader) > 0, "empty test loader"

    if save_model_and_dataloader:
        torch.save(test_loader, os.path.join("test_loader", f"loader_{save_name}"))

    return train_loader, val_loader, test_loader

