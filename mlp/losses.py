# Loss functions.

from abc import ABC, abstractmethod

import numpy as np

# A tiny number which is used to keep probabilities away from 0 or 1, as
# log(0) is negative infinity and would break the optimizer's comparisons.
_EPS = 1e-12  # keeps log() away from 0 (spec: protect against p ~ 0 or 1)

class Loss(ABC):
    # Blueprint: every loss must turn (true labels, predictions) into ONE number.
    @abstractmethod
    def compute(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        # Return a scalar loss value.
        pass
    
def _align(y_true, y_pred):
    # Make y_true the same shape as y_pred. Without this, labels of shape (n,)
    # against predictions of shape (n, 1) would silently broadcast to (n, n).
    y_true = np.asarray(y_true, dtype=float)
    if y_true.size != y_pred.size:
        raise ValueError(f"Shape mismatch: y_true {y_true.shape} vs y_pred {y_pred.shape}")
    return y_true.reshape(y_pred.shape)

class MSE(Loss):
    # Mean squared error: average of (true - predicted) squared.
    def compute(self, y_true, y_pred):
        y_true = _align(y_true, y_pred)
        return float(np.mean((y_true - y_pred) ** 2))
    
class BinaryCrossEntropy(Loss):
    # Loss for two-class problems. A confident wrong answer is punished heavily.
    def compute(self, y_true, y_pred):
        y_true = _align(y_true, y_pred)
        p = np.clip(y_pred, _EPS, 1.0 - _EPS)  # keep p strictly inside (0, 1)
        # -[y*ln(p) + (1-y)*ln(1-p)], averaged over all rows
        return float(-np.mean(y_true * np.log(p) + (1.0 - y_true) * np.log(1.0 - p)))
    
class CategoricalCrossEntropy(Loss):
    # Expects one-hot y_true and probability rows in y_pred.
    def compute(self, y_true, y_pred):
        y_true = _align(y_true, y_pred)
        p = np.clip(y_pred, _EPS, 1.0)
        # Sum over classes (axis=1) for each row, then average over rows.
        return float(-np.mean(np.sum(y_true * np.log(p), axis=1)))