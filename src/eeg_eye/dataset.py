from loguru import logger
import numpy as np
import torch
from pathlib import Path

from eeg_eye.data import load, DEFAULT_DATA_DIR


def _segments(X: np.ndarray, y: np.ndarray) -> list[tuple[np.ndarray, int]]:
    split_indices = np.where(np.diff(y) != 0)[0] + 1
    x_segments = np.split(X, split_indices)
    y_segments = np.split(y, split_indices)

    return [(x_seg, int(y_seg[0])) for x_seg, y_seg in zip(x_segments, y_segments)]


class EEGEyeDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir: Path = DEFAULT_DATA_DIR):
        X, y = load(data_dir=data_dir)
        self.segments = _segments(X, y)

    def __len__(self):
        return len(self.segments)

    def __getitem__(self, idx: int):
        x_segment, label = self.segments[idx]
        return torch.tensor(x_segment, dtype=torch.float32), torch.tensor(
            label, dtype=torch.float32
        )
