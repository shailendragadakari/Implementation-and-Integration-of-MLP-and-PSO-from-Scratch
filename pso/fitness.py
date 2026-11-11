# FitnessFunction interface. PSO minimizes the returned scalar.
# This is the only thing PSO knows about the problem it is solving: "give me a list
# of numbers, and I will give you one score back; lower is better".
# PSO never imports anything from the neural-network code, which keeps the two parts independent.

from abc import ABC, abstractmethod
import numpy as np

class FitnessFunction(ABC):
    @abstractmethod
    def evaluate(self, position: np.ndarray) -> float:
        # Return a scalar fitness for a candidate solution; smaller is better.
        pass