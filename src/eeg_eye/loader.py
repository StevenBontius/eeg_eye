import torch
from eeg_eye.dataset import EEGEyeDataset
from pathlib import Path
from typing import Literal


def get_dataloader(
    data_dir: Path,
    batch_size: int,
    window_size: int,
    stride: int,
    split_ratio: float,
    split: Literal["train", "val"],
) -> torch.utils.data.DataLoader:
    data_set = EEGEyeDataset(
        window_size=window_size,
        stride=stride,
        data_dir=data_dir,
        split=split,
        split_ratio=split_ratio,
    )

    return torch.utils.data.DataLoader(
        dataset=data_set, batch_size=batch_size, shuffle=True
    )
