# Shared helper: build the MLP, couple it to PSO via the adapter, and train.
# Used by the GUI, main.py's steps and the experiments.

from adapter import MLPFitnessAdapter, MLPFitnessAdapterWithActivations
from mlp import MLPBuilder
from pso import NearestNeighbourInformantSelection, Population, RandomInformantSelection

ACTIVATION_CHOICES = ("sigmoid", "tanh", "relu")


def train_pso_mlp(X_train, y_train, config, callback=None):
    # config keys: hidden, activations, strategy, k, particles, iterations,
    #     alpha, beta, delta, gamma, epsilon, seed.
    # Optional keys (all default to the plain algorithm):
    #     boundary ("none"|"clamp"|"reflect"|"random"), bound, inertia_schedule ("constant"|"linear"),
    #     inertia_end, evolve_activations (True -> PSO also chooses the hidden activations).
    # Returns (network, loss, population).
    seed = config["seed"]
    # 1) Build the network (sizes derived from the data) and its loss.
    builder = MLPBuilder(seed=seed).set_dataset(X_train, y_train)
    builder.set_configuration({"hidden_layers": config["hidden"],
                               "activations": config["activations"],
                               "task": "binary_classification"})
    network, loss = builder.build()
    # 2) Couple network and PSO through the adapter (scores a particle on training data).
    if config.get("evolve_activations", False):
        fitness = MLPFitnessAdapterWithActivations(network, loss, X_train, y_train,
                                                   choices=ACTIVATION_CHOICES)
        dimension = fitness.dimension
    else:
        fitness = MLPFitnessAdapter(network, loss, X_train, y_train)
        dimension = network.count_parameters()
    # 3) Choose how particles pick neighbours.
    strategy = (NearestNeighbourInformantSelection() if config["strategy"] == "nearest"
                else RandomInformantSelection(seed=seed))
    # 4) Create the swarm; each particle is a full set of network parameters.
    population = Population(
        fitness_function=fitness, informant_strategy=strategy,
        num_particles=config["particles"], num_informants=config["k"],
        num_iterations=config["iterations"], dimension=dimension,
        inertia_weight=config["alpha"], cognitive_coefficient=config["beta"],
        social_coefficient=config["delta"], global_coefficient=config["gamma"],
        step_size=config["epsilon"], seed=seed,
        boundary=config.get("boundary", "none"), bound=config.get("bound", 5.0),
        inertia_schedule=config.get("inertia_schedule", "constant"),
        inertia_end=config.get("inertia_end", 0.4),
    )
    # 5) Run PSO, then load the best solution found into the network.
    best_position, _ = population.optimise(callback=callback)
    fitness.install(best_position)
    return network, loss, population
