import typing
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.multiprocessing as mp

from lib.utils.devtools import performance
from lib.utils.logger import Logger


def evaluate_batch(args):
    """Evaluate a batch in a multiprocessing pool."""
    models, X, y, loss_fn = args
    with torch.no_grad():
        losses = torch.stack([loss_fn(model(X)[:, :-1], y[:, 1:]) for model in models], dim=1)
    return losses


class MLPLEvaluator:
    def __init__(self, loss: nn.Module, dataloader: DataLoader, progress_interval: int = 100):
        self.__loss = loss
        self.__dataloader = dataloader
        self.__progress_interval = progress_interval

    def __get_data_size(self):
        if hasattr(self.__dataloader, "dataset"):
            return len(self.__dataloader.dataset)
        return len(self.__dataloader) * self.__dataloader.batch_size

    @performance.track_func_performance()
    def __start_evaluators(self, models: typing.List[nn.Module]) -> typing.List[torch.Tensor]:
        Logger.info("Starting evaluators...")
        pool = mp.Pool(mp.cpu_count())
        tasks = [
            (models, X, y, self.__loss)
            for i, (X, y) in enumerate(self.__dataloader)
        ]
        results = pool.map(evaluate_batch, tasks)
        pool.close()
        pool.join()
        Logger.info(f"Evaluated {len(results)} batches...")
        return results

    @performance.track_func_performance()
    def __collapse_losses(self, losses: torch.Tensor) -> torch.Tensor:
        return torch.min(losses, dim=1).values

    @performance.track_func_performance()
    def evaluate(self, models: typing.List[nn.Module]) -> float:
        mp.set_start_method("spawn", force=True)
        for model in models:
            model.eval()

        results = self.__start_evaluators(models)
        data_size = self.__get_data_size()
        losses = torch.zeros((data_size, len(models)))

        for i, batch_losses in enumerate(results):
            start_idx = i * self.__dataloader.batch_size
            end_idx = start_idx + self.__dataloader.batch_size
            losses[start_idx:end_idx] = batch_losses

            if i % self.__progress_interval == 0:
                Logger.info(f"Evaluating: {i} / {len(self.__dataloader)}...")

        losses = self.__collapse_losses(losses)
        return torch.mean(losses)
