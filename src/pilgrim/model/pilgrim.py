"""
Pilgrim is a distance prediction model on a graph.

It was introduced here https://github.com/khoruzhii/cayleypy-cube.
"""

from typing import Any

import torch
import torch.nn.functional as F  # noqa: N812
from torch import nn

from .model_blocks import ResidualBlock


class Pilgrim(nn.Module):
    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()
        self.dtype = torch.float32
        self.state_size = config["state_size"]
        self.num_classes = config["num_classes"]
        self.hd1 = config["hd1"]
        self.hd2 = config["hd2"]
        self.nrd = config["nrd"]
        self.z_add = 0
        self.output_dim = 1

        state_size = self.state_size
        num_classes = self.num_classes
        hd1 = self.hd1
        hd2 = self.hd2
        nrd = self.nrd

        self.input_layer = nn.Linear(state_size * num_classes, hd1)
        self.bn1 = nn.BatchNorm1d(hd1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(config["dropout_rate"])

        if hd2 > 0:
            self.hidden_layer = nn.Linear(hd1, hd2)
            self.bn2 = nn.BatchNorm1d(hd2)
            hidden_dim_for_output = hd2
        else:
            self.hidden_layer = None
            self.bn2 = None
            hidden_dim_for_output = hd1

        if nrd > 0 and hd2 > 0:
            self.residual_blocks = nn.ModuleList([
                ResidualBlock(hd2, config["dropout_rate"]) for _ in range(nrd)
            ])
        else:
            self.residual_blocks = None

        self.output_layer = nn.Linear(hidden_dim_for_output, self.output_dim)

    def forward(self, z):
        # One-hot encode and flatten to dense
        x = (
            F
            .one_hot(z.long() + self.z_add, num_classes=self.num_classes)
            .view(z.size(0), -1)
            .to(self.dtype)
        )

        # Input block
        x = self.input_layer(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout(x)

        # Optional hidden block
        if self.hidden_layer is not None:
            x = self.hidden_layer(x)
            x = self.bn2(x)
            x = self.relu(x)
            x = self.dropout(x)

        # Optional residual stack
        if self.residual_blocks is not None:
            for block in self.residual_blocks:
                x = block(x)

        # Output
        x = self.output_layer(x)
        return x.flatten()
