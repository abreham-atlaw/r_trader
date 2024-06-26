import typing
import torch
import torch.nn as nn
from core.utils.research.model.layers import Indicators
from core.utils.research.model.model.cnn.residual_block import ResidualBlock
from core.utils.research.model.model.linear.model import LinearModel
from core.utils.research.model.model.savable import SavableModule


class ResNet(SavableModule):
    def __init__(
            self,
            num_classes: int,
            extra_len: int,
            conv_channels: typing.List[int],
            kernel_sizes: typing.List[int],
            ff_linear: LinearModel = None,
            indicators: typing.Optional[Indicators] = None,
            pool_sizes: typing.Optional[typing.List[int]] = None,
            hidden_activation: typing.Optional[nn.Module] = None,
            dropout_rates: typing.Union[typing.List[float], float] = 0,
            ff_dropout: float = 0,
            init_fn: typing.Optional[nn.Module] = None,
            padding: int = 1,
            avg_pool=True,
            linear_collapse=False,
            input_size: int = 1028,
            norm: typing.Union[bool, typing.List[bool]] = False,
    ):
        super(ResNet, self).__init__()
        self.args = {
            'extra_len': extra_len,
            'ff_linear': ff_linear,
            'num_classes': num_classes,
            'conv_channels': conv_channels,
            'kernel_sizes': kernel_sizes,
            'pool_sizes': pool_sizes,
            'hidden_activation': hidden_activation.__class__.__name__ if hidden_activation else None,
            'init_fn': init_fn.__name__ if init_fn else None,
            'dropout_rates': dropout_rates,
            'padding': padding,
            'avg_pool': avg_pool,
            'linear_collapse': linear_collapse,
            'input_size': input_size,
            'norm': norm,
            'indicators': indicators
        }
        self.extra_len = extra_len
        self.layers = nn.ModuleList()
        self.pool_layers = nn.ModuleList()
        self.norm_layers = nn.ModuleList()
        self.dropout_layers = nn.ModuleList()
        self.input_size = input_size

        if isinstance(dropout_rates, int):
            dropout_rates = [dropout_rates for _ in conv_channels]

        if len(dropout_rates) != len(conv_channels):
            raise ValueError("Dropout size doesn't match layers size")

        if indicators is None:
            indicators = Indicators()
        self.indicators = indicators

        if pool_sizes is None:
            pool_sizes = [0 for _ in kernel_sizes]
        conv_channels = [self.indicators.indicators_len] + conv_channels

        if isinstance(norm, bool):
            norm = [norm for _ in range(len(conv_channels) - 1)]
        if len(norm) != len(conv_channels) - 1:
            raise ValueError("Norm size doesn't match layers size")

        for i in range(len(conv_channels) - 1):
            self.layers.append(
                ResidualBlock(
                    in_channels=conv_channels[i],
                    out_channels=conv_channels[i + 1],
                    kernel_size=kernel_sizes[i],
                    padding=padding,
                    hidden_activation=hidden_activation,
                    init_fn=init_fn,
                    dropout=dropout_rates[i],
                    norm=norm[i],
                    pool_size=pool_sizes[i]
                )
            )
            if pool_sizes[i] > 0:
                if avg_pool:
                    pool = nn.AvgPool1d(kernel_size=pool_sizes[i], stride=2)
                else:
                    pool = nn.MaxPool1d(kernel_size=pool_sizes[i], stride=2)
                self.pool_layers.append(pool)
            else:
                self.pool_layers.append(nn.Identity())

            if dropout_rates[i] > 0:
                self.dropout_layers.append(nn.Dropout(dropout_rates[i])
                )
            else:
                self.dropout_layers.append(nn.Identity())

        self.hidden_activation = hidden_activation
        if ff_linear is None:
            self.fc = nn.Linear(conv_channels[-1] + self.extra_len, num_classes)
        else:
            self.fc = nn.Sequential(
                nn.Linear(conv_channels[-1] + self.extra_len, ff_linear.input_size),
                ff_linear,
                nn.Linear(ff_linear.output_size, num_classes)
            )

        if ff_dropout > 0:
            self.ff_dropout = nn.Dropout(ff_dropout)
        else:
            self.ff_dropout = nn.Identity()

        self.ff_linear = ff_linear
        self.fc_layer = None
        self.num_classes = num_classes
        self.collapse_layer = None if linear_collapse else nn.AdaptiveAvgPool1d((1,))
        self.__init()

    def __init(self):
        init_data = torch.rand((1, self.input_size))
        self(init_data)

    def collapse(self, out: torch.Tensor) -> torch.Tensor:
        return torch.flatten(out, 1, 2)

    def fc(self, out: torch.Tensor) -> torch.Tensor:
        if self.fc_layer is None:
            if self.ff_linear is None:
                self.fc_layer = nn.Linear(out.shape[-1], self.num_classes)
            else:
                self.fc_layer = nn.Sequential(
                    nn.Linear(out.shape[-1], self.ff_linear.input_size),
                    self.ff_linear,
                    nn.Linear(self.ff_linear.output_size, self.num_classes)
                )
        return self.fc_layer(out)

    def forward(self, x):
        seq = x[:, :-self.extra_len]
        out = self.indicators(seq)
        for layer, pool_layer, dropout in zip(self.layers, self.pool_layers, self.dropout_layers):
            out = layer(out)
            out = pool_layer(out)
            out = dropout(out)
        out = self.collapse(out)
        out = out.reshape(out.size(0), -1)
        out = self.ff_dropout(out)
        out = torch.cat((out, x[:, -self.extra_len:]), dim=1)
        out = self.fc(out)
        return out

    def export_config(self) -> typing.Dict[str, typing.Any]:
        return self.args
