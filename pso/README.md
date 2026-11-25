# pso: particle swarm optimisation from scratch (NumPy only)

A generic optimiser that minimises a scalar fitness. It contains nothing specific to neural networks and does not import `mlp/` (a test checks this).

| File | Contents |
|---|---|
| `fitness.py` | `FitnessFunction` interface: `evaluate(position)` returns a scalar, smaller is better |
| `particle.py` | `Particle`: position, velocity, personal best, informants, velocity and position updates |
| `informants.py` | `InformantSelectionStrategy` (Strategy pattern) with `RandomInformantSelection` and `NearestNeighbourInformantSelection` |
| `population.py` | `Population`: the swarm, the global best and the main loop (Algorithm 39), plus optional boundary handling and an inertia schedule |

Comments marked `Alg. 39 line N` refer to Algorithm 39 in *Essentials of Metaheuristics* (Luke), Section 3.5. Names of the coefficients: alpha = inertia, beta = personal best, delta = informants' best, gamma = global best, epsilon = step size. This swaps the book's gamma and delta; see the header of `particle.py`.
