import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from eeg_eye.config import TrainingConfig
from eeg_eye.utils import get_device
from loguru import logger


class Trainer:
    def __init__(
        self, model: nn.Module, dataloader: DataLoader, config: TrainingConfig
    ):
        self.device = get_device()
        self.model = model
        self.model.to(self.device)
        self.dataloader = dataloader
        self.config = config
        self.optimizer = optim.Adam(
            params=self.model.parameters(), lr=config.learning_rate
        )
        self.loss_fn = nn.BCELoss()

    def train(self):
        best_loss = float("inf")
        patience_counter = 0

        for epoch in range(self.config.epochs):
            loss, accu = self._train_epoch()
            logger.info(f"Epoch {epoch}:Train loss: {loss} Train accuracy: {accu}")

            # early stopping
            if loss < best_loss:
                best_loss = loss
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= self.config.patience:
                logger.info(f"Stopping early at epoch {epoch}")
                break

    def _train_epoch(self) -> tuple[float, float]:
        total_loss = 0
        total_accuracy = 0
        for x, y in self.dataloader:
            x = x.to(self.device)
            y = y.to(self.device)

            self.optimizer.zero_grad()
            y_hat = self.model(x).squeeze(-1)
            loss = self.loss_fn(y_hat, y)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            # calculating accuracy
            predicted = (y_hat > 0.5).float()
            accuracy = (predicted == y).float().mean()
            total_accuracy += accuracy

        avg_loss = total_loss / len(self.dataloader)
        avg_acc = total_accuracy / len(self.dataloader)

        return (avg_loss, avg_acc)

    def _evaluate(self):
        None
