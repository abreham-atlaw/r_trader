import torch
import torch.nn as nn


class CNN(nn.Module):

    def __init__(self, num_classes, conv_channels, kernel_sizes, pool_sizes, dropout_rate):
        super(CNN, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(len(conv_channels) - 1):
            self.layers.append(nn.Sequential(
                nn.Conv1d(in_channels=conv_channels[i], out_channels=conv_channels[i+1], kernel_size=kernel_sizes[i], stride=1, padding=1),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=pool_sizes[i], stride=2),
                nn.Dropout(dropout_rate)
            ))
        self.avg_pool = nn.AdaptiveAvgPool1d((1,))
        self.fc = nn.Linear(conv_channels[-1], num_classes)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        out = torch.unsqueeze(x, dim=1)
        for layer in self.layers:
            out = layer(out)
        out = self.avg_pool(out)
        out = out.reshape(out.size(0), -1)
        out = self.dropout(out)
        out = self.fc(out)
        return out
