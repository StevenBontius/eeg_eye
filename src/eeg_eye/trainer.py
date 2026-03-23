import torch
import torch.nn as nn
import torch.optim as optim
from torchmetrics.classification import BinaryAccuracy
from torch.utils.data import DataLoader
from eeg_eye.config import TrainingConfig
from eeg_eye.utils import get_device
from loguru import logger


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        test_loader: DataLoader,
        config: TrainingConfig,
    ):
        self.device = get_device()
        self.model = model
        self.model.to(self.device)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.config = config
        self.optimizer = optim.Adam(
            params=self.model.parameters(), lr=config.learning_rate
        )
        self.loss_fn = nn.BCELoss()
        self.accuracy = BinaryAccuracy().to(self.device)

    def train(self):
        best_test_loss = float("inf")
        patience_counter = 0

        for epoch in range(self.config.epochs):
            train_loss, train_accu = self._train_epoch()
            test_loss, test_accu = self._test_epoch()

            logger.info(
                f"Epoch {epoch}: Train loss: {train_loss:.4f} accuracy: {train_accu:.4f} -> Test loss: {test_loss:.4f} accuracy {test_accu:.4f}"
            )

            # early stopping
            if test_loss < best_test_loss:
                best_test_loss = test_loss
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= self.config.patience:
                logger.info(f"Stopping early at epoch {epoch}")
                break

    def _train_epoch(self) -> tuple[float, float]:
        self.model.train()
        total_loss = 0
        for x, y in self.train_loader:
            x = x.to(self.device)
            y = y.to(self.device)

            self.optimizer.zero_grad()
            y_hat = self.model(x).squeeze(-1)
            loss = self.loss_fn(y_hat, y)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            self.accuracy(y_hat, y)

        avg_loss = total_loss / len(self.train_loader)
        avg_acc = self.accuracy.compute()
        self.accuracy.reset()

        return (avg_loss, avg_acc)

    def _test_epoch(self) -> tuple[float, float]:
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for x, y in self.test_loader:
                x = x.to(self.device)
                y = y.to(self.device)

                y_hat = self.model(x).squeeze(-1)
                loss = self.loss_fn(y_hat, y)
                total_loss += loss.item()
                self.accuracy(y_hat, y)

        avg_loss = total_loss / len(self.test_loader)
        avg_acc = self.accuracy.compute()
        self.accuracy.reset()

        return (avg_loss, avg_acc)
