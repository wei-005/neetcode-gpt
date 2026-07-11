import numpy as np
from typing import List


class Solution:
    def forward_and_backward(self,
                              x: List[float],
                              W1: List[List[float]], b1: List[float],
                              W2: List[List[float]], b2: List[float],
                              y_true: List[float]) -> dict:
        # Architecture: x -> Linear(W1, b1) -> ReLU -> Linear(W2, b2) -> predictions
        # Loss: MSE = mean((predictions - y_true)^2)
        #
        # Return dict with keys:
        #   'loss':  float (MSE loss, rounded to 4 decimals)
        #   'dW1':   2D list (gradient w.r.t. W1, rounded to 4 decimals)
        #   'db1':   1D list (gradient w.r.t. b1, rounded to 4 decimals)
        #   'dW2':   2D list (gradient w.r.t. W2, rounded to 4 decimals)
        #   'db2':   1D list (gradient w.r.t. b2, rounded to 4 decimals)
        # pass

        x = np.array(x, dtype=float)
        W1 = np.array(W1, dtype=float)
        b1 = np.array(b1, dtype=float)
        W2 = np.array(W2, dtype=float)
        b2 = np.array(b2, dtype=float)
        y_true = np.array(y_true, dtype=float)

        result = {}

        z1 = x @ W1.T + b1
        a1 = np.maximum(0, z1)
        z2 = a1 @ W2.T + b2
        y_hat = z2

        loss = np.mean((y_true - y_hat) ** 2)

        dz2 = 2 * (z2 - y_true) / y_true.size

        # dW2 = dz2 * a1.T
        dW2 = np.outer(dz2, a1)

        db2 = dz2

        # da1 = W2.T * dz2
        da1 = W2.T @ dz2
        
        relu_mask = (z1 > 0).astype(float)

        dz1 = da1 * relu_mask

        # dW1 = dz1 * x.T
        dW1 = np.outer(dz1, x)

        db1 = dz1

        result['loss'] = np.round(loss, 4)
        result['dW1'] = np.round(dW1, 4)
        result['db1'] = np.round(db1, 4)
        result['dW2'] = np.round(dW2, 4)
        result['db2'] = np.round(db2, 4)

        return result
