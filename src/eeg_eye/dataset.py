from loguru import logger
import numpy as np
import torch
from pathlib import Path
from typing import Literal

from eeg_eye.data import load, DEFAULT_DATA_DIR


def _segments(X: np.ndarray, y: np.ndarray) -> list[tuple[np.ndarray, int]]:
    split_indices = np.where(np.diff(y) != 0)[0] + 1
    x_segments = np.split(X, split_indices)
    y_segments = np.split(y, split_indices)

    return [(x_seg, int(y_seg[0])) for x_seg, y_seg in zip(x_segments, y_segments)]


class EEGEyeDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        split: Literal["train", "val"] = "train",
        window_size: int = 50,
        stride: int = 50,
        data_dir: Path = DEFAULT_DATA_DIR,
        split_ratio: float = 0.8,
        seed: int = 42,
    ):
        np.random.seed(seed)
        X, y = load(data_dir=data_dir)
        self.segments = _segments(X, y)

        # splitting
        np.random.shuffle(self.segments)
        n_train = int(len(self.segments) * split_ratio)
        if split == "train":
            self.segments = self.segments[:n_train]
        else:
            self.segments = self.segments[n_train:]

        self.window_size = window_size
        self.stride = stride
        # calculate window sizes
        self.window_count = [
            max(0, (len(x) - self.window_size) // self.stride) for x, _ in self.segments
        ]
        # calculate cumulative window count for indexing, include 0 for first segment
        self.cumulative_count = np.concatenate([[0], np.cumsum(self.window_count)])

    def __len__(self):
        return sum(self.window_count)

    def __getitem__(self, idx: int):
        # find segment, one step back for added
        segment = np.searchsorted(self.cumulative_count, idx, side="right") - 1
        # get the segment
        x_array, label = self.segments[segment]
        # find the index of the window within the segment
        segment_index = idx - self.cumulative_count[segment]
        # calculate where the window should start
        window_start = segment_index * self.stride
        # get the window from the segment
        x_window = x_array[window_start : window_start + self.window_size]
        # return tensors
        return torch.tensor(x_window, dtype=torch.float32), torch.tensor(
            label, dtype=torch.float32
        )
