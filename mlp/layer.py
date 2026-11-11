# Layer class. Holds weights, biases and an activation object.
# It never optimizes or changes its own parameters.

import numpy as np

class Layer:
    def __init__(self, n_in: int, n_out: int, activation, rng=None):
        self.n_in = n_in            # how many numbers come into this layer
        self.n_out = n_out          # how many neurons this layer has
        self.activation = activation
        # rng = random number generator (passed in so runs can be repeatable).
        rng = rng if rng is not None else np.random.default_rng()
        # Glorot/Xavier-style start so initial outputs are not saturated.
        limit = np.sqrt(6.0 / (n_in + n_out))
        # One weight per (input, neuron) pair -> a table of shape (n_in, n_out).
        self.weights = rng.uniform(-limit, limit, size=(n_in, n_out))
        self.biases = np.zeros(n_out)  # one bias per neuron, starting at 0

    def forward(self, X: np.ndarray) -> np.ndarray:
        # X has one row per data sample. "@" is matrix multiplication: it does
        # every neuron's weighted sum for every sample at once. Then add biases
        # and apply the activation.
        return self.activation.forward(X @ self.weights + self.biases)

    def count_parameters(self) -> int:
        return self.n_in * self.n_out + self.n_out  # weights + biases

    def get_parameters(self) -> np.ndarray:
        # Order: weights (row-major flatten), then biases.
        return np.concatenate([self.weights.ravel(), self.biases])

    def set_parameters(self, parameters: np.ndarray) -> None:
        if parameters.size != self.count_parameters():
            raise ValueError(
                f"Expected {self.count_parameters()} parameters, got {parameters.size}"
            )
        n_w = self.n_in * self.n_out  # how many of the numbers are weights
        # First n_w numbers -> weights table; the rest -> biases.
        # .copy() so later changes to the caller's list cannot alter this layer.
        self.weights = parameters[:n_w].reshape(self.n_in, self.n_out).copy()
        self.biases = parameters[n_w:].copy()