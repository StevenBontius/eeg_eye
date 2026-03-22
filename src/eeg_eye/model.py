import torch.nn as nn
from eeg_eye.config import ModelConfig


class GRUmodel(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()

        self.gru = nn.GRU(
            input_size=config.input_size,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            batch_first=True,
        )

        self.linear = nn.Linear(in_features=config.hidden_size, out_features=1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        gru_output, _ = self.gru(x)
        last_step = gru_output[:, -1, :]
        lin_output = self.linear(last_step)
        return self.sigmoid(lin_output)