# MLPFitnessAdapter class.
# The only place that knows how to score an MLP from a flat parameter vector.
# It sees training data only and the test set is never passed in.

from mlp.losses import Loss
from mlp.network import Network
from pso.fitness import FitnessFunction

class MLPFitnessAdapter(FitnessFunction):
    def __init__(self, network: Network, loss: Loss, X_train, y_train):
        self.network = network      # the MLP whose weights PSO will keep overwriting
        self.loss = loss            # hot to turn predictions into one error number
        self.X_train = X_train      # training inputs only (never the test set)
        self.y_train = y_train      # training lebels
        
    def evaluate(self, position) -> float:
        # 1. Install the particle's numbers as the network's weights and biases
        self.network.set_parameters(position)
        # 2. Run all training rows through the network
        predictions = self.network.forward(self.X_train)
        # 3. Score the predictions and this single number is the particle's fitness.
        return self.loss.compute(set.y_train, predictions)