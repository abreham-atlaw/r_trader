import torch
import typing

from torch.utils.data import DataLoader
from tqdm import tqdm

from core.utils.research.training.callbacks import Callback


class Trainer:

    def __init__(self, model, loss_function, optimizer, callbacks: typing.List[Callback]=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if torch.cuda.device_count() > 1:
            print("Found use", torch.cuda.device_count(), "GPUs.")
            model = torch.nn.DataParallel(model)
        if callbacks is None:
            callbacks = []
        self.model = model.to(self.device)
        self.loss_function = loss_function
        self.optimizer = optimizer
        self.callbacks = callbacks

    def summary(self):
        print("Model Summary")
        print("Layer Name" + "\t" * 7 + "Number of Parameters")
        print("=" * 100)
        total_params = 0
        for name, parameter in self.model.named_parameters():
            if not parameter.requires_grad:
                continue
            param = parameter.numel()
            total_params += param
            print(name + "\t" * 3 + str(param))
        print("=" * 100)
        print(f"Total Params:{total_params}")

    def train(
            self,
            dataloader: DataLoader,
            val_dataloader=None,
            epochs: int = 1,
            progress: bool = False,
            shuffle=True,
            progress_interval=100
    ):
        for callback in self.callbacks:
            callback.on_train_start(self.model)
        self.summary()

        train_losses = []
        val_losses = []
        for epoch in range(epochs):
            if shuffle:
                dataloader.dataset.shuffle()
            for callback in self.callbacks:
                callback.on_epoch_start(self.model, epoch)
            self.model.train()
            running_loss = 0.0
            pbar = tqdm(dataloader) if progress else dataloader
            for i, (X, y) in enumerate(pbar):
                for callback in self.callbacks:
                    callback.on_batch_start(self.model, i)
                X, y = X.to(self.device), y.to(self.device)
                self.optimizer.zero_grad()
                y_hat = self.model(X)
                loss = self.loss_function(y_hat, y)
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()
                if progress and i % progress_interval == 0:
                    pbar.set_description(f"Epoch {epoch + 1} loss: {running_loss / (i + 1)}")
                for callback in self.callbacks:
                    callback.on_batch_end(self.model, i)
                i += 1
            epoch_loss = running_loss / len(dataloader)
            print(f"Epoch {epoch + 1} completed, loss: {epoch_loss}")
            train_losses.append(epoch_loss)
            for callback in self.callbacks:
                callback.on_epoch_end(self.model, epoch)

            if val_dataloader is not None:
                val_loss = self.validate(val_dataloader)
                val_losses.append(val_loss)
                print(f"Validation loss: {val_loss}")
        for callback in self.callbacks:
            callback.on_train_end(self.model)
        return train_losses, val_losses

    def validate(self, dataloader):
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for X, y in dataloader:
                X, y = X.to(self.device), y.to(self.device)
                y_hat = self.model(X)
                loss = self.loss_function(y_hat, y)
                total_loss += loss.item()
        return total_loss / len(dataloader)
