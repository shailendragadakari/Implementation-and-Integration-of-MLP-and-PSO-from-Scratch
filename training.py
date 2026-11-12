# Shared helper: build the MLP, couple it to PSO vis the adapter and, train.
# Used by the GUI: mirrors main.py's steps.

from adapter import MLPFitnessAdapter
from mlp import MLPBuilder
from pso import NearestNeighbourInformantSelection, Population, RandomInformantSelection


def train_pso_mlp(X_train, y_train, config, callback=None):
    # config keys: hidden, activations, strategy, k, particles, iterations,
    #     alpha, beta, delta, gamma, epsilon, seed. Returns (network, loss, population).
    seed = config["seed"]
    # 1) Build the network (sizes derived from the data) and its loss.
    builder = MLPBuilder(seed=seed).set_dataset(X_train, y_train)
    builder.set_configuration({"hidden_layers": config["hidden"],
                               "activations": config["activations"],
                               "task": "binary_classification"})
    network, loss = builder.build()
    # 2) Couple network and PSO through the adapter (scores a particle on training data).
    fitness = MLPFitnessAdapter(network, loss, X_train, y_train)
    # 3) Choose how particles pick neighbours.
    strategy = (NearestNeighbourInformantSelection() if config["strategy"] == "nearest"
                else RandomInformantSelection(seed=seed))
    # 4) Create the swarm; each particle is a full set of network parameters.
    population = Population(
        fitness_function=fitness, informant_strategy=strategy,
        num_particles=config["particles"], num_informants=config["k"],
        num_iterations=config["iterations"], dimension=network.count_parameters(),
        inertia_weight=config["alpha"], cognitive_coefficient=config["beta"],
        social_coefficient=config["delta"], global_coefficient=config["gamma"],
        step_size=config["epsilon"], seed=seed,
    )
    # 5) Run PSO, then load the best parameters found into the network.
    best_position, _ = population.optimise(callback=callback)
    network.set_parameters(best_position)
    return network, loss, population