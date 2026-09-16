import torch
import torch.nn as nn

from src.config import HIDDEN_SIZE, NEURAL_TICKS, INPUT_SIZE


class FlyBrain(nn.Module):

    def __init__(self, input_size=INPUT_SIZE, output_size=4):
        super().__init__()

        self.input_layer = nn.Linear(input_size, HIDDEN_SIZE)
        # self.hidden_layer = nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE)
        self.hidden_layer1 = nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE)
        self.hidden_layer2 = nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE)
        self.output_layer = nn.Linear(HIDDEN_SIZE, output_size)

        self.neural_ticks = NEURAL_TICKS
        self.threshold = 1.0
        self.membrane_decay = 0.85

    def forward(self, x):
        voltage = torch.zeros(
            x.size(0),
            HIDDEN_SIZE,
            device=x.device,
        )

        spike_sum = torch.zeros_like(voltage)

        for _ in range(self.neural_ticks):
            current = self.input_layer(x)
            # current = current + self.hidden_layer(voltage)
            current = current + self.hidden_layer1(voltage)

            voltage = (
                voltage * self.membrane_decay
                + current
            )

            spike = torch.sigmoid(
                5.0 * (voltage - self.threshold)
            )

            voltage = voltage * (1.0 - spike)

            current = self.hidden_layer2(voltage)

            voltage = (
                voltage * self.membrane_decay
                + current
            )

            spike = torch.sigmoid(
                2.0 * (voltage - self.threshold)
            )

            voltage = voltage * (1.0 - spike)

            spike_sum += spike

        hidden_activity = spike_sum / self.neural_ticks

        q_values = self.output_layer(hidden_activity)

        return q_values
