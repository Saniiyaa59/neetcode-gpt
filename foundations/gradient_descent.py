class Solution:
    def get_minimizer(self, iterations: int, learning_rate: float, init: int) -> float:
        # Objective function: f(x) = x^2
        # Derivative:         f'(x) = 2x
        # Update rule:        x = x - learning_rate * f'(x)
        # Round final answer to 5 decimal places
        x_old = init
        x_new = init
        for i in range (iterations):
            dy_dx = 2 * x_old
            x_new = x_old - learning_rate * dy_dx
            x_old = x_new

        return round(x_new,5)

