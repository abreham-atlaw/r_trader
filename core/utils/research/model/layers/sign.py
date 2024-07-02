import torch
import torch.nn as nn


class Sign(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, inputs, *args, **kwargs):
        return torch.sign(inputs)


class SignFilter(nn.Module):
    def __init__(self, sign, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__sign = sign
        self.__sign_layer = Sign()

    def forward(self, inputs, *args, **kwargs):
        return torch.abs(
            torch.round(
                (self.__sign_layer(inputs) + self.__sign)/2
            )
        )*inputs
