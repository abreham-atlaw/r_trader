import torch
import torch.nn as nn


class CNN(nn.Module):
    def __init__(self, num_classes, conv_channels, kernel_sizes):
        super(CNN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv1d(in_channels=conv_channels[0], out_channels=conv_channels[1], kernel_size=kernel_sizes[0], stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv1d(in_channels=conv_channels[1], out_channels=conv_channels[2], kernel_size=kernel_sizes[1], stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )
        self.avgpool = nn.AdaptiveAvgPool1d((1))
        self.fc = nn.Linear(conv_channels[2], num_classes)

    def forward(self, x):
        out = self.layer1(torch.unsqueeze(x, dim=1))
        out = self.layer2(out)
        out = self.avgpool(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        return out
