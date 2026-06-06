import numpy as np
from numpy.typing import NDArray


class Solution:
    def get_derivative(self, model_prediction: NDArray[np.float64], ground_truth: NDArray[np.float64], N: int, X: NDArray[np.float64], desired_weight: int) -> float:
        # note that N is just len(X)
        return -2 * np.dot(ground_truth - model_prediction, X[:, desired_weight]) / N

    def get_model_prediction(self, X: NDArray[np.float64], weights: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.squeeze(np.matmul(X, weights))

    learning_rate = 0.01

    def train_model(
        self,
        X: NDArray[np.float64],
        Y: NDArray[np.float64],
        num_iterations: int,
        initial_weights: NDArray[np.float64]
    ) -> NDArray[np.float64]:

        #X is Nx3
        #Y is Nx1
        weights = initial_weights
        N = X.shape[0]
        # For each iteration:
        for i in range(num_iterations):
            #   1. Compute predictions with get_model_prediction(X, weights)
            pred_y = self.get_model_prediction(X, weights)
            #   2. For each weight index j, compute gradient with get_derivative()
            for j in range (3):
                gradient = self.get_derivative(pred_y, Y, N, X, j)
                #   3. Update: weights[j] -= learning_rate * gradient
                weights[j] -= self.learning_rate * gradient

        # Return np.round(final_weights, 5)
        return np.round(weights, 5)
