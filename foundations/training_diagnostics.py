import torch
import torch.nn as nn
from typing import List, Dict


class Solution:

    def compute_activation_stats(self, model: nn.Module, x: torch.Tensor) -> List[Dict[str, float]]:
        # Forward pass through model layer by layer
        # After each nn.Linear, record: mean, std, dead_fraction
        # Run with torch.no_grad(). Round to 4 decimals.
        # pass
        stats = []
        h = x

        # model.eval()

        with torch.no_grad():
            for layer in model:
                h = layer(h)

                if isinstance(layer, nn.Linear):
                    mean = h.mean().item()
                    std = h.std().item()
                    dead = (h<=0).all(dim=0)
                    dead_fraction = dead.float().mean().item()

                    stats.append({'mean': round(mean, 4), 'std': round(std, 4), 'dead_fraction': round(dead_fraction, 4)})

        return stats


    def compute_gradient_stats(self, model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> List[Dict[str, float]]:
        # Forward + backward pass with nn.MSELoss
        # For each nn.Linear layer's weight gradient, record: mean, std, norm
        # Call model.zero_grad() first. Round to 4 decimals.
        # pass
        stats = []
        model.zero_grad()
        h = x

        for layer in model:
            h = layer(h)

        y_hat = h

        loss_fn = nn.MSELoss()
        loss = loss_fn(y_hat, y)

        loss.backward()

        for layer in model:
            if isinstance(layer, nn.Linear):
                grad = layer.weight.grad

                mean = grad.mean().item()
                std = grad.std().item()
                norm = torch.norm(grad, p=2).item()

                stats.append({'mean': round(mean, 4), 'std': round(std, 4), 'norm': round(norm, 4)})

        return stats



    def diagnose(self, activation_stats: List[Dict[str, float]], gradient_stats: List[Dict[str, float]]) -> str:
        # Classify network health based on the stats
        # Return: 'dead_neurons', 'exploding_gradients', 'vanishing_gradients', or 'healthy'
        # Check in priority order (see problem description for thresholds)
        # pass
        if any(s['dead_fraction'] > 0.5 for s in activation_stats):
            return 'dead_neurons'
        elif any(s['norm'] > 1000 for s in gradient_stats):
            return 'exploding_gradients'
        elif gradient_stats[-1]['norm'] < 1e-5:
            return 'vanishing_gradients'
        elif any(s['std'] < 0.1 for s in activation_stats):
            return 'vanishing_gradients'
        elif any(s['std'] > 10 for s in activation_stats):
            return 'exploding_gradients'
        else:
            return 'healthy'
