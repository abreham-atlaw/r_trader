
import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        padding,
        hidden_activation,
        init_fn=None,
        norm=True,
        res_norm=True,
        pool_size=0,
        dropout=0
    ):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=padding)
        self.norm1 = nn.BatchNorm1d(out_channels) if norm else nn.Identity()
        self.pool1 = nn.AvgPool1d(kernel_size=pool_size, stride=2) if pool_size > 0 else nn.Identity()
        self.activation = hidden_activation
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size, padding=padding)
        self.norm2 = nn.BatchNorm1d(out_channels) if norm else nn.Identity()
        self.res_norm = nn.BatchNorm1d(out_channels) if res_norm else nn.Identity()
        self.init_fn = init_fn
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        if in_channels != out_channels:
            self.shortcut = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        else:
            self.shortcut = nn.Identity()

        if init_fn is not None:
            init_fn(self.conv1.weight)
            init_fn(self.conv2.weight)
            if isinstance(self.shortcut, nn.Conv1d):
                init_fn(self.shortcut.weight)

    def residual(self, identity, out):
        identity = self.pool1(identity)  # Apply the same pooling operation to the identity tensor
        start = (identity.size(2) - out.size(2)) // 2
        end = identity.size(2) - start
        out = out + identity[:, :, start:end]
        out = self.res_norm(out)
        return out

    def forward(self, x):
        identity = self.shortcut(x)
        out = self.norm1(x)
        out = self.conv1(out)
        out = self.activation(out)
        out = self.pool1(out)  # Apply the pooling operation to the output tensor

        out = self.dropout(out)
        out = self.norm2(out)
        out = self.conv2(out)
        out = self.activation(out)

        out = self.residual(identity, out)

        return out
