# MLPBuilder class
# Input size, output size, output activation, and loss are derived from the dataset 
# and the explicitly-given task type; the user only supplies hidden layers/activations.

import numpy as np

from mlp.activations import Sigmoid, Softmax, get_activation
from mlp.layer import Layer
from mlp.losses import BinaryCrossEntropy, CategoricalCrossEntropy, Loss
from mlp.network import Network

TASKS = ("binary_classification", "multiclass_classification")

class MLPBuilder:
    def __init__(self, seed=None):
        self.X = None
        self.y = None
        self.config = None
        self._rng = np.random.default_rng(seed)  # seed -> repeatable starting weights
        self._network = None
        self._loss = None

    def set_dataset(self, X, y):
        self.X = np.asarray(X)
        self.y = np.asarray(y)
        return self  # returning self lets calls be chained: builder.set_...().set_...()

    def set_configuration(self, config: dict):
        # Check the recipe before building anything, with clear error messages.
        for key in ("hidden_layers", "activations", "task"):
            if key not in config:
                raise ValueError(f"Configuration is missing '{key}'")
        if len(config["hidden_layers"]) != len(config["activations"]):
            raise ValueError(
                "Number of activations must equal number of hidden layers "
                f"({len(config['activations'])} vs {len(config['hidden_layers'])})"
            )
        if any(int(h) < 1 for h in config["hidden_layers"]):
            raise ValueError("Hidden layer sizes must be positive integers")
        if config["task"] not in TASKS:
            raise ValueError(f"task must be one of {TASKS}")
        for name in config["activations"]:
            get_activation(name)  # raises if unknown
        self.config = config
        return self

    def determine_input_size(self) -> int:
        return int(self.X.shape[1])  # one input per column (feature) of the data

    def determine_output_size(self) -> int:
        if self.config["task"] == "binary_classification":
            return 1  # a single probability
        return int(len(np.unique(self.y)))  # one output per class

    def determine_output_activation(self):
        if self.config["task"] == "binary_classification":
            return Sigmoid()  # one number between 0 and 1
        return Softmax()      # probabilities across classes

    def determine_loss(self) -> Loss:
        if self.config["task"] == "binary_classification":
            return BinaryCrossEntropy()
        return CategoricalCrossEntropy()
    
    def build_layers(self) -> Network:
        network = Network()
        # Layer sizes in order, for example [4, 8, 5]: input layer size, then each hidden layer size.
        sizes = [self.determine_input_size()] = [int(h) for h in self.config["hidden_layers"]]
        # Pair up neighbours: (4->8 with act 1), (8->5 with act 2) and so on...
        for n_in, n_out, act in zip(sizes[:1], sizes[1:], self.config["activations"]):
            network.add_layer(Layer(n_in, n_out, get_activation(act), self._rng))
        # Finally the output layer, whose size and activation come from the task.
        network.add_layer(
            Layer(sizes[-1], self.determine_output_size(),
                  self.determine_output_activation(), self._rng)
        )
        return network
    
    def build(self):
        if self.X is None or self.y is None:
            raise RuntimeError("Call set_dataset() and set_configuration() before build()")
        self._network = self.build_layers()
        self._loss = self.determine_loss()
        return self._network, self._loss    # the network and the loss to score it with 