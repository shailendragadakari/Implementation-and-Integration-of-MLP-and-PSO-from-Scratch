# Network class. Ordered layers, forward pass and flat parameter vector.
# No training algorithm stays here.

import numpy as np

from mlp.layer import Layer


class Network:
    def __init__(self):
        self.layers: list[Layer] = []  # layers in order: first hidden ... output

    def add_layer(self, layer: Layer) -> None:
        self.layers.append(layer)

    def forward(self, X: np.ndarray) -> np.ndarray:
        out = X
        for layer in self.layers:
            out = layer.forward(out)  # each layer's output feeds the next layer
        return out

    def predict(self, X: np.ndarray) -> np.ndarray:
        # Return probabilities/outputs. Converting to labels is a separate step.
        return self.forward(X)

    def count_parameters(self) -> int:
        # This is D, the length of the vector PSO searches over.
        return sum(layer.count_parameters() for layer in self.layers)

    def get_parameters(self) -> np.ndarray:
        # All layers' weights and biases joined into one long list.
        return np.concatenate([layer.get_parameters() for layer in self.layers])

    def set_parameters(self, parameters: np.ndarray) -> None:
        parameters = np.asarray(parameters, dtype=float).ravel()
        if parameters.size != self.count_parameters():
            raise ValueError(
                f"Expected {self.count_parameters()} parameters, got {parameters.size}"
            )
        start = 0
        for layer in self.layers:
            # Hand each layer its own slice of the long list, in order.
            end = start + layer.count_parameters()
            layer.set_parameters(parameters[start:end])
            start = end