# Dataset loading, stratified splitting and standardisation (NumPy only).
import numpy as np


def load_dataset(path):
    # Banknote authentication: 4 numeric features + binary label in last column.
    data = np.loadtxt(path, delimiter=",")  # whole file -> table of numbers
    return data[:, :-1], data[:, -1].astype(int)


def train_test_split(X, y, test_size=0.2, random_state=42, stratify=True):
    # Stratified split: each class is split with the same ratio.
    rng = np.random.default_rng(random_state)  # fixed seed -> same split every time
    test_idx = []
    # Group row numbers by class so each class is split separately (stratified).
    groups = [np.where(y == c)[0] for c in np.unique(y)] if stratify else [np.arange(len(y))]
    for idx in groups:
        idx = rng.permutation(idx)  # shuffle rows within this class
        # The first test_size fraction of the shuffled rows go to the test set.
        test_idx.extend(idx[: int(round(len(idx) * test_size))])
    # Mark test rows True; everything else is training.
    test_mask = np.zeros(len(y), dtype=bool)
    test_mask[test_idx] = True
    # Shuffle again so classes are mixed rather than grouped together.
    train_idx = rng.permutation(np.where(~test_mask)[0])
    test_idx = rng.permutation(np.where(test_mask)[0])
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


class Standardiser:
    # z-score scaling. fit() on training data only, then transform() everything.

    def fit(self, X):
        self.mean_ = X.mean(axis=0)   # average of each column
        self.std_ = X.std(axis=0)     # spread of each column
        self.std_[self.std_ == 0] = 1.0  # avoid dividing by zero for a constant column
        return self

    def transform(self, X):
        # z-score: how many standard deviations from the mean (uses TRAINING mean/std).
        return (X - self.mean_) / self.std_

    def fit_transform(self, X):
        return self.fit(X).transform(X)
