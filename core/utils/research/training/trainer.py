
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

import uuid
import typing

from core.utils.research.training.callbacks import Callback
from core.utils.research.training.data.state import TrainingState


class Trainer:

    def __init__(
            self,
            model,
            cls_loss_function=None,
            reg_loss_function=None,
            optimizer=None,
            callbacks: typing.List[Callback]=None,
            max_norm: typing.Optional[float] = None,
            clip_value: typing.Optional[float] = None,
            log_gradient_stats: bool = False
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if torch.cuda.device_count() > 1:
            print("Found use", torch.cuda.device_count(), "GPUs.")
            model = torch.nn.DataParallel(model)
        if callbacks is None:
            callbacks = []
        self.model = self.__initialize_model(model)
        self.cls_loss_function = cls_loss_function
        self.reg_loss_function = reg_loss_function
        self.optimizer = optimizer
        self.callbacks = callbacks
        self.__state: typing.Optional[TrainingState] = None
        self.__max_norm = max_norm
        self.__clip_value = clip_value
        self.__log_gradient_stats = log_gradient_stats

    @property
    def state(self) -> typing.Optional[TrainingState]:
        return self.__state

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

    @staticmethod
    def __get_default_state() -> TrainingState:
        return TrainingState(
            id=uuid.uuid4().hex,
            epoch=0,
            batch=0
        )

    def __initialize_model(self, model: nn.Module) -> nn.Module:
        init_data = torch.rand((1, model.input_size))
        model = model.to(torch.device("cpu"))
        model(init_data)
        return model.to(self.device).float()

    @staticmethod
    def __split_y(y: torch.Tensor) -> typing.Tuple[torch.Tensor, torch.Tensor]:
        return y[:, :-1], y[:, -1:]

    def __loss(self, y_hat, y):
        cls_y, reg_y = self.__split_y(y)
        cls_y_hat, reg_y_hat = self.__split_y(y_hat)

        cls_loss = self.cls_loss_function(cls_y_hat, cls_y)
        reg_loss = self.reg_loss_function(reg_y_hat, reg_y)

        loss = cls_loss + reg_loss
        return cls_loss, reg_loss, loss

    def __format_loss(self, loss):
        return f"loss: {loss[2]}(cls: {loss[0]}, reg: {loss[1]})"

    def train(
            self,
            dataloader: DataLoader,
            val_dataloader=None,
            epochs: int = 1,
            progress: bool = False,
            shuffle=True,
            progress_interval=100,
            cls_loss_only=False,
            reg_loss_only=False,
            state: typing.Optional[TrainingState] = None
    ):
        if self.optimizer is None or self.cls_loss_function is None:
            raise ValueError("Model not setup(optimizer or loss function missing")

        if state is None:
            state = self.__get_default_state()

        self.__state = state

        for callback in self.callbacks:
            callback.on_train_start(self.model)
        self.summary()

        if cls_loss_only:
            print("Training Classifier")

        if reg_loss_only:
            print("Training Regressor")

        train_losses = []
        val_losses = []
        for epoch in range(state.epoch, epochs):
            min_gradient = float('inf')
            max_gradient = float('-inf')
            state.epoch = epoch
            if shuffle:
                dataloader.dataset.shuffle()
            for callback in self.callbacks:
                callback.on_epoch_start(self.model, epoch)
            self.model.train()
            running_loss = torch.zeros((3,))
            pbar = tqdm(dataloader) if progress else dataloader
            for i, (X, y) in enumerate(pbar):
                if i < state.batch:
                    continue
                state.batch = i
                for callback in self.callbacks:
                    callback.on_batch_start(self.model, i)
                X, y = X.to(self.device).type(X.type()), y.to(self.device).type(y.type())
                self.optimizer.zero_grad()
                y_hat = self.model(X)

                cls_loss, ref_loss, loss = self.__loss(y_hat, y)
                if cls_loss_only:
                    cls_loss.backward()
                elif reg_loss_only:
                    ref_loss.backward()
                else:
                    loss.backward()

                if self.__log_gradient_stats is not None:
                    for param in self.model.parameters():
                        if param.grad is not None:
                            grad_data = param.grad.data
                            min_gradient = min(min_gradient, grad_data.min().item())
                            max_gradient = max(max_gradient, grad_data.max().item())

                if self.__max_norm is not None:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.__max_norm)

                if self.__clip_value is not None:
                    torch.nn.utils.clip_grad_value_(self.model.parameters(), clip_value=self.__clip_value)

                self.optimizer.step()
                running_loss += torch.FloatTensor([l.item() for l in [cls_loss, ref_loss, loss]])
                if progress and i % progress_interval == 0:
                    pbar.set_description(f"Epoch {epoch + 1} {self.__format_loss(running_loss/(i+1))}")
                for callback in self.callbacks:
                    callback.on_batch_end(self.model, i)
            state.batch = 0
            epoch_loss = (running_loss / len(dataloader)).tolist()
            print(f"Epoch {epoch + 1} completed, {self.__format_loss(epoch_loss)}")
            if self.__log_gradient_stats:
                print(f"Min gradient: {min_gradient}, Max gradient: {max_gradient}")

            train_losses.append(epoch_loss)
            losses = (epoch_loss,)

            if val_dataloader is not None:
                val_loss = self.validate(val_dataloader)
                val_losses.append(val_loss)
                print(f"Validation loss: {self.__format_loss(val_loss)}")
                losses = (epoch_loss, val_loss)

            for callback in self.callbacks:
                callback.on_epoch_end(self.model, epoch, losses)

        for callback in self.callbacks:
            callback.on_train_end(self.model)
        return train_losses, val_losses

    def validate(self, dataloader):
        self.model.eval()
        total_loss = torch.zeros((3,))
        with torch.no_grad():
            for X, y in dataloader:
                X, y = X.to(self.device), y.to(self.device)
                y_hat = self.model(X)
                cls_loss, ref_loss, loss = self.__loss(y_hat, y)

                total_loss += torch.FloatTensor([l.item() for l in [cls_loss, ref_loss, loss]])
        return (total_loss / len(dataloader)).tolist()
