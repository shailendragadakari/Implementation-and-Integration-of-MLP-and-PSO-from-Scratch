# Particle. One particle = one candidate solution that moves around the search space.
#
# "Alg. 39 line N" comments refer to Algorithm 39 (PSO with informants) in Luke,
# "Essentials of Metaheuristics", Section 3.5, as numbered in the Week 7 lecture slides.
# NOTE: double-check these line numbers against your copy of the book / slides.
# Naming used here (follows the coursework design document): alpha = inertia,
# beta = personal best, delta = informants' best, gamma = global best, epsilon = step size.

import numpy as np

from pso.fitness import FitnessFunction


class Particle:
    def __init__(self, position, velocity, rng=None):
        self.position = np.asarray(position, dtype=float)  # where it is now (a list of D numbers)
        self.velocity = np.asarray(velocity, dtype=float)  # how it is moving (same length)
        # Personal best = the best place this particle has ever visited.
        self.best_position = self.position.copy()
        self.best_fitness = np.inf   # infinity = "nothing seen yet", so any real score beats it
        self.fitness = np.inf        # score at the current position
        self.informants: list["Particle"] = []  # the neighbours it listens to
        # Each particle has its own random generator for its r1, r2, r3 numbers.
        self._rng = rng if rng is not None else np.random.default_rng()

    def evaluate(self, fitness_function: FitnessFunction) -> None:
        # Alg. 39 line 13: AssessFitness(x)
        # Score the current position (for the network: run the data, get the loss).
        self.fitness = float(fitness_function.evaluate(self.position))

    def update_personal_best(self) -> None:
        # Alg. 39 line 17: x* <- previous fittest location of x (the particle's own best).
        # Lower is better, so "<" means an improvement. Copy the position so the
        # remembered best is not changed when the particle moves on.
        if self.fitness < self.best_fitness:
            self.best_fitness = self.fitness
            self.best_position = self.position.copy()

    def get_informants_best(self):
        # Alg. 39 line 18: x+ <- previous fittest location of the informants of x.
        # Best personal-best among this particle's informants -> (position, fitness).
        if not self.informants:
            raise RuntimeError("Particle has no informants; call assign_informants() first")
        best = min(self.informants, key=lambda p: p.best_fitness)  # lowest score wins
        return best.best_position, best.best_fitness

    def update_velocity(self, informants_best, global_best, alpha, beta, gamma, delta):
        # Alg. 39 lines 19-22: x! <- global best location; for each dimension i draw
        #         b, c, d uniformly from 0..beta, 0..delta, 0..gamma (here: coefficient * U[0,1))
        #         and set v_i <- alpha*v_i + b*(x*_i - x_i) + c*(x+_i - x_i) + d*(x!_i - x_i).
        # v <- alpha*v + beta*r1*(pbest-x) + delta*r2*(informants_best-x) + gamma*r3*(gbest-x)
        #
        #         r1, r2, r3 are drawn per dimension, uniformly in [0, 1).
        #         Naming follows the spec's formula: alpha=inertia, beta=cognitive,
        #         delta=informant (social), gamma=global-best.
        d = self.position.size  # number of dimensions
        # Fresh random numbers each update, one per dimension, for each of the 3 pulls.
        r1, r2, r3 = self._rng.random(d), self._rng.random(d), self._rng.random(d)
        self.velocity = (
            alpha * self.velocity                                    # keep some of the old direction (inertia)
            + beta * r1 * (self.best_position - self.position)       # pull toward my own best
            + delta * r2 * (informants_best - self.position)         # pull toward my neighbours' best
            + gamma * r3 * (global_best - self.position)             # pull toward the swarm's best (0 by default)
        )

    def update_position(self, epsilon: float = 1.0) -> None:
        # Alg. 39 line 24: x <- x + epsilon * v
        # Move: new position = old position + step size * velocity.
        self.position = self.position + epsilon * self.velocity