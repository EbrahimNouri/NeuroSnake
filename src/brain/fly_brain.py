import torch
import torch.nn as nn

from src.config import INPUT_SIZE, HIDDEN_SIZE


class FlyBrain(nn.Module):

    def __init__(self, input_size=INPUT_SIZE, output_size=4):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_size, HIDDEN_SIZE),
            nn.ReLU(),
            nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE),
            nn.ReLU(),
            nn.Linear(HIDDEN_SIZE, output_size),
        )

    def forward(self, x):
        return self.net(x)