from dataclasses import dataclass


@dataclass
class ModelConfig:
    input_size: int = 14
    hidden_size: int = 64
    num_layers: int = 1

@dataclass
class TrainingConfig:
    batch_size: int = 32
    epochs: int = 25
    learning_rate: float = 0.001
    patience: int = 2