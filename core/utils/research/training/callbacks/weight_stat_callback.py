import torch
from .callback import Callback


class WeightStatsCallback(Callback):

    def __init__(self):
        self.__batch_store = []

    def on_batch_end(self, model, batch, logs=None):
        min_val, max_val, min_abs_val = float('inf'), float('-inf'), float('inf')

        for param in model.parameters():
            if param.requires_grad:
                min_val = min(min_val, torch.min(param.data).item())
                max_val = max(max_val, torch.max(param.data).item())
                min_abs_val = min(min_abs_val, torch.min(torch.abs(param.data)).item())

        self.__batch_store.append((min_val, max_val, min_abs_val))

    def on_epoch_end(self, model, epoch, losses, logs=None):
        min_val = min([values[0] for values in self.__batch_store])
        max_val = max([values[1] for values in self.__batch_store])
        min_abs_val = min([values[2] for values in self.__batch_store])

        print(f'After epoch {epoch}, min weight: {min_val}, max weight: {max_val}, min abs weight: {min_abs_val}')
