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
        x = np.array(x).reshape(-1,1) #Nx1
        W1 = np.array(W1) #HxN
        b1 = np.array(b1).reshape(-1,1) #Hx1
        W2 = np.array(W2) #
        b2 = np.array(b2).reshape(-1,1)
        y_true = np.array(y_true).reshape(-1,1)

        #forward
        z1 = W1 @ x + b1
        a1 = np.maximum(0, z1)
        z2 = W2 @ a1 + b2
        #Loss
        loss = np.mean((z2 - y_true)**2)

        #backward
        n = len(y_true)
        dL_dz2 = 2 * (z2 - y_true) / n 
        dz2_dw2 = a1 
        dz2_db2 = 1
        dz2_da1 = W2 
        da1_dz1 = (z1 > 0).astype(float) 
        dz1_dw1 = x 
        dz1_db1 = 1

        dL_dw2 = dL_dz2 @ dz2_dw2.T
        dL_db2 = dL_dz2

        dL_dz1 = (dz2_da1.T @ dL_dz2) * da1_dz1 
        dL_dw1 = dL_dz1 @ dz1_dw1.T
        dL_db1 = dL_dz1

        return {
            'loss': round(float(loss), 4),
            'dW1': np.round(dL_dw1, 4).tolist(),
            'db1': np.round(dL_db1.flatten(), 4).tolist(),
            'dW2': np.round(dL_dw2, 4).tolist(),
            'db2': np.round(dL_db2.flatten(), 4).tolist(),
        }
