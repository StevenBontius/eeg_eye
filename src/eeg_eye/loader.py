import torch
from eeg_eye.dataset import EEGEyeDataset
from eeg_eye.data import DEFAULT_DATA_DIR
from pathlib import Path
from torch.nn.utils.rnn import pad_sequence
import numpy as np


def _collate_fn(batch: list) -> tuple[torch.Tensor, torch.Tensor]:
    x, y = zip(*batch)

    padded_x = pad_sequence(sequences=x, batch_first=True)
    y = torch.stack(y)
    return (padded_x, y)


def get_dataloader(
    data_dir: Path = DEFAULT_DATA_DIR, batch_size: int = 4
) -> torch.utils.data.DataLoader:
    data_set = EEGEyeDataset(data_dir)

    return torch.utils.data.DataLoader(
        dataset=data_set, batch_size=batch_size, collate_fn=_collate_fn
    )
