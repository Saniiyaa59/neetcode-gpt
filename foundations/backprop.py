import numpy as np
from numpy.typing import NDArray
from typing import Tuple


class Solution:
    def backward(self, x: NDArray[np.float64], w: NDArray[np.float64], b: float, y_true: float) -> Tuple[NDArray[np.float64], float]:
        # x: 1D input array
        # w: 1D weight array
        # b: scalar bias
        # y_true: true target value
        #
        # Forward: z = dot(x, w) + b, y_hat = sigmoid(z)
        # Loss: L = 0.5 * (y_hat - y_true)^2
        # Return: (dL_dw rounded to 5 decimals, dL_db rounded to 5 decimals)
        def sigmoid(z):
            return 1/(1+np.exp(-z))

        z = x.T @ w + b #1x1
        y_hat = sigmoid(z) #1x1

        dL_dyhat = (y_hat - y_true) #1x1
        dyhat_dz = y_hat * (1-y_hat) #1x1
        dz_dw = x #Nx1
        dz_db = 1

        dL_dz = dL_dyhat * dyhat_dz #1x1

        dL_dw = dL_dz * dz_dw
        dL_db = dL_dz * dz_db

        return (np.round(dL_dw, 5), np.round(dL_db, 5))
