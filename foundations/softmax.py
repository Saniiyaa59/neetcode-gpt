import numpy as np
from numpy.typing import NDArray


class Solution:

    def softmax(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        # z is a 1D NumPy array of logits
        # Hint: subtract max(z) for numerical stability before computing exp
        # return np.round(your_answer, 4)
        z_new = z - np.max(z)
        numerator = np.exp(z_new)
        denominator = np.sum(np.exp(z_new))
        return np.round(numerator/denominator, 4)

        
