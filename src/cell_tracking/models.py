import torch
import torch.nn as nn


class Simple3DCellDetector(nn.Module):
    def __init__(self):
        super().__init__()

        self.model = nn.Sequential(
            nn.Conv3d(1, 8, kernel_size=3, padding=1),
            nn.ReLU(),

            nn.Conv3d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(),

            nn.Conv3d(16, 8, kernel_size=3, padding=1),
            nn.ReLU(),

            nn.Conv3d(8, 1, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.model(x)