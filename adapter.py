# MLPFitnessAdapter class.
# The only place that knows how to score an MLP from a flat parameter vector.
# It sees training data only and the test set is never passed in.

import numpy as np

from mlp.activations import get_activation
from mlp.losses import Loss
from mlp.network import Network
from pso.fitness import FitnessFunction


class MLPFitnessAdapter(FitnessFunction):
    def __init__(self, network: Network, loss: Loss, X_train, y_train):
        self.network = network      # the MLP whose weights PSO will keep overwriting
        self.loss = loss            # how to turn predictions into one error number
        self.X_train = X_train      # training inputs only (never the test set)
        self.y_train = y_train      # training labels

    def install(self, position) -> None:
        # Put a candidate solution into the network (also used to install PSO's final answer).
        self.network.set_parameters(position)

    def evaluate(self, position) -> float:
        # 1) Install the particle's numbers as the network's weights and biases.
        self.install(position)
        # 2) Run all training rows through the network.
        predictions = self.network.forward(self.X_train)
        # 3) Score the predictions; this single number is the particle's fitness.
        return self.loss.compute(self.y_train, predictions)


class MLPFitnessAdapterWithActivations(MLPFitnessAdapter):
    # Extension: PSO also searches over the hidden-layer ACTIVATION FUNCTIONS.
    #
    # The particle vector is  [ all weights and biases | one extra gene per hidden layer ].
    # Each extra gene is a real number. It is squashed with tanh into (-1, 1), rescaled to
    # (0, 1) and cut into equal bins, one bin per allowed activation, so ANY real value
    # decodes to a valid choice (no bounds are needed on these genes).
    # PSO itself is unchanged: it still just sees a longer vector and a scalar fitness.

    def __init__(self, network, loss, X_train, y_train, choices=("sigmoid", "tanh", "relu")):
        super().__init__(network, loss, X_train, y_train)
        self.choices = tuple(choices)
        self.n_hidden = len(network.layers) - 1
        self.n_weights = network.count_parameters()  # weights/biases only (before the extra genes)

    @property
    def dimension(self) -> int:
        # Length of the vector PSO must search: parameters + one gene per hidden layer.
        return self.n_weights + self.n_hidden

    def decode_activations(self, position) -> list:
        # Turn the trailing genes of a position into a list of activation names.
        genes = np.asarray(position, dtype=float)[self.n_weights:]
        unit = (np.tanh(genes) + 1.0) / 2.0                    # (0, 1)
        idx = np.minimum((unit * len(self.choices)).astype(int), len(self.choices) - 1)
        return [self.choices[i] for i in idx]

    def install(self, position) -> None:
        position = np.asarray(position, dtype=float)
        if position.size != self.dimension:
            raise ValueError(f"Expected {self.dimension} values, got {position.size}")
        self.network.set_hidden_activations(
            [get_activation(n) for n in self.decode_activations(position)])
        self.network.set_parameters(position[:self.n_weights])