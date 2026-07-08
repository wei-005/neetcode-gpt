import numpy as np
from numpy.typing import NDArray


class Solution:
    
    def sigmoid(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        # z is a 1D NumPy array
        # Formula: 1 / (1 + e^(-z))
        # return np.round(your_answer, 5)
        res = []
        for element in z:
            element = round(1 / (1 + math.exp(-element)), 5)
            res.append(element)
        return res

    def relu(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        # z is a 1D NumPy array
        # Formula: max(0, z) element-wise
        res = []
        for element in z:
            res.append(round(max(0.0, element), 5))
        return res
