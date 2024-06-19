
import torch.nn as nn


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, hidden_activation, init_fn=None, norm=True):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=padding)
        self.norm1 = nn.BatchNorm1d(out_channels) if norm else nn.Identity()
        self.activation = hidden_activation
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size, padding=padding)
        self.norm2 = nn.BatchNorm1d(out_channels) if norm else nn.Identity()
        self.init_fn = init_fn

        if in_channels != out_channels:
            self.shortcut = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        else:
            self.shortcut = nn.Identity()

        if init_fn is not None:
            init_fn(self.conv1.weight)
            init_fn(self.conv2.weight)
            if isinstance(self.shortcut, nn.Conv1d):
                init_fn(self.shortcut.weight)

    def forward(self, x):
        identity = self.shortcut(x)

        out = self.conv1(x)
        out = self.norm1(out)
        out = self.activation(out)

        out = self.conv2(out)
        out = self.norm2(out)

        out += identity
        out = self.activation(out)

        return out
