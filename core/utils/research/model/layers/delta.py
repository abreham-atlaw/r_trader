import torch.nn as nn


class Delta(nn.Module):

    def forward(self, inputs):
        return inputs[:, 1:] - inputs[:, :-1]
