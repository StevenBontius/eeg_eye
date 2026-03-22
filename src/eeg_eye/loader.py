import torch
from eeg_eye.dataset import EEGEyeDataset
from eeg_eye.data import DEFAULT_DATA_DIR
from pathlib import Path


def get_dataloader(
    data_dir: Path = DEFAULT_DATA_DIR, batch_size: int = 4, window_size=50, stride=50
) -> torch.utils.data.DataLoader:
    data_set = EEGEyeDataset(window_size=window_size, stride=stride, data_dir=data_dir)

    return torch.utils.data.DataLoader(dataset=data_set, batch_size=batch_size)
