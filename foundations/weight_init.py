import torch
import torch.nn as nn
import math
from typing import List


class Solution:

    def xavier_init(self, fan_in: int, fan_out: int) -> List[List[float]]:
        # Return a (fan_out x fan_in) weight matrix using Xavier/Glorot normal initialization
        # Use torch.manual_seed(0) for reproducibility
        # Round to 4 decimal places and return as nested list
        # pass

        torch.manual_seed(0)

        std = math.sqrt(2 / (fan_in + fan_out))

        return torch.round(torch.randn(fan_out, fan_in) * std, decimals=4).tolist()

    def kaiming_init(self, fan_in: int, fan_out: int) -> List[List[float]]:
        # Return a (fan_out x fan_in) weight matrix using Kaiming/He normal initialization (for ReLU)
        # Use torch.manual_seed(0) for reproducibility
        # Round to 4 decimal places and return as nested list
        # pass
        torch.manual_seed(0)

        std = math.sqrt(2 / (fan_in))

        return torch.round(torch.randn(fan_out, fan_in) * std, decimals=4).tolist()

    def check_activations(self, num_layers: int, input_dim: int, hidden_dim: int, init_type: str) -> List[float]:
        # Forward random input through num_layers with the given init_type.
        # Use torch.manual_seed(0) once at the start.
        # Return the std of activations after each layer, rounded to 2 decimals.
        # pass
        torch.manual_seed(0)

        # h = torch.randn(input_dim)

        weights = []

        # stds = []

        for i in range(num_layers):
            fan_in = input_dim if i == 0 else hidden_dim
            fan_out = hidden_dim

            if init_type == 'xavier':
                std = math.sqrt(2 / (fan_in + fan_out))
                W = torch.randn(fan_out, fan_in) * std
            elif init_type == 'kaiming':
                std = math.sqrt(2 / fan_in)
                W = torch.randn(fan_out, fan_in) * std
            else:
                W = torch.randn(fan_out, fan_in)

            weights.append(W)

        h = torch.randn(input_dim)

        stds = []

        for W in weights:
            h = W @ h
            h = torch.relu(h)
            stds.append(round(h.std().item(), 2))

        return stds

            

