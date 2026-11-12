# Activation functions for MLPs. Each exposes only forward(z): the network is never trained by back-propagation, so no derivatives are needed.
# The network is never trained by back-propagation, so no derivatives are needed.
from abc import ABC, abstractmethod

import numpy as np


class ActivationFunction(ABC):
    # ABC = "abstract base class": a blueprint that cannot be used directly.
    # Every real activation below must provide its own forward() method.
    @abstractmethod
    def forward(self, z: np.ndarray) -> np.ndarray:
        # Apply the activation element-wise (or row-wise for softmax).
        pass


class Sigmoid(ActivationFunction):
    # Squashes any number into the range 0..1:  1 / (1 + e^-z).

    def forward(self, z):
        # Numerically stable: never exponentiates a large positive number.
        # (e^1000 would overflow, so positive and negative z use different but
        # mathematically equal formulas.)
        out = np.empty_like(z, dtype=float)  # empty array to fill in
        pos = z >= 0                         # True where z is zero or positive
        out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))   # safe: exponent is <= 0
        ez = np.exp(z[~pos])                        # safe: z is negative here
        out[~pos] = ez / (1.0 + ez)
        return out


class Tanh(ActivationFunction):
    # Squashes any number into the range -1..1, centred on 0.

    def forward(self, z):
        return np.tanh(z)


class Softmax(ActivationFunction):
    # Turns each row of scores into probabilities that add up to 1.
    #     Only needed for problems with more than two classes.

    def forward(self, z):
        # Subtracting each row's maximum changes nothing mathematically but
        # keeps e^x from overflowing.
        shifted = z - np.max(z, axis=1, keepdims=True)  # stability
        e = np.exp(shifted)
        return e / np.sum(e, axis=1, keepdims=True)  # each row sums to 1


# Lookup table: the word used in a configuration -> the class to create.
_REGISTRY = {"sigmoid": Sigmoid, "tanh": Tanh, "softmax": Softmax}


def get_activation(name: str) -> ActivationFunction:
    # Create an activation object from its name (used by the builder).
    try:
        return _REGISTRY[name.lower()]()  # look up the class, then create one
    except KeyError:
        # Unknown word: fail loudly and list the valid choices.
        raise ValueError(f"Unknown activation '{name}'. Options: {sorted(_REGISTRY)}")