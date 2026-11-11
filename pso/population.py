# Population class. Owns the particles, global best and the main loop.
# The Greek-letter names follow the specification: alpha = inertia, beta = cognitive pull,
# delta = informant pull, gamma = global best pull, epsilon = step size.

import numpy as np
from pso.fitness import FitnessFunction
from pso.informants import InformantSelectionStrategy
from pso.particle import Particle

class Population:
    def __init__(
        self,
        fitness_function: FitnessFunction,
        informant_strategy: InformantSelectionStrategy,
        num_particles: int,
        num_informants: int,
        num_iterations: int,
        dimension: int,
        inertia_weight: float = 0.7,        # alpha
        cognitive_coefficient: float = 1.5,  # beta  (personal best)
        social_coefficient: float = 1.5,     # delta (best informant)
        global_coefficient: float = 0.0,     # gamma (global best)
        step_size: float = 1.0,              # epsilon
        init_range: float = 1.0,
        patience=None,
        seed=None,
    ):
        # Validate early so mistakes give a clear message instead of a confusing crash later.
        if not 1 <= num_informants <= num_particles - 1:
            raise ValueError("Need 1 <= num_informants <= num_particles - 1")
        self.fitness_function = fitness_function
        self.informant_strategy = informant_strategy
        self.num_particles = num_particles
        self.num_informants = num_informants
        self.num_iterations = num_iterations
        self.dimension = dimension
        self.alpha = inertia_weight
        self.beta = cognitive_coefficient
        self.delta = social_coefficient
        self.gamma = global_coefficient
        self.epsilon = step_size
        self.init_range = init_range
        self.patience = patience  # stop if global best unchanged this many iterations
        self._rng = np.random.default_rng(seed)  # seed -> repeatable runs
        
        self.particles: list[Particle] = []
        self.global_best_position = None   # best position found by ANY particle so far
        self.global_best_fitness = np.inf
        # Convergence record: index 0 = after initial evaluation, then one per iteration.
        self.history = {"global_best": [], "mean_fitness": []}
        self.n_evaluations = 0  # how many times the fitness function has been
        
    # pseudocode steps
    def initialise(self) -> None:
        r = self.init_range
        # Scatter particles uniformly in [-r, r] in every dimension. Starting
        # velocities are small (10% of that range) so early moves are gentle.
        self.particles = [
            Particle(
                position=self._rng.uniform(-r, r, size=self.dimension),
                velocity=self._rng.uniform(-r, r, size=self.dimension) * 0.1,  # small initial velocity
                rng=np.random.default_rng(self._rng.integers(2**32)),    # each particle gets its own generator
            )
            for _ in range(self.num_particles)
        ]
        self.global_best_position = None
        self.global_best_fitness = np.inf
        self.history = {"global_best": [], "mean_fitness": []}
        self.n_evaluations = 0
        
    def evaluate_particles(self) -> None:
        # Score each particle's current position and update its personal best.
        for p in self.particles:
            p.evaluate(self.fitness_function)
        self.n_evaluations += len(self.particles)
        
    def update_personal_bests(self) -> None:
        for p in self.particles:
            p.update_personal_best()
            
    def update_global_best(self) -> None:
        # Find the best personal-best among all particles.
        best = min(self.particles, key=lambda p: p.best_fitness)
        if best.best_fitness < self.global_best_fitness:
            self.global_best_fitness = best.best_fitness
            self.global_best_position = best.best_position.copy()
            
    def assign_informants(self) -> None:
        # Each particle gets a list of k other particles to listen to.
        for p in self.particles:
            p.informants = self.informant_strategy.select_informants(
                p, self.particles, self.num_informants
            )
            
    def _record(self) -> None:
        self.history["global_best"].append(self.global_best_fitness)
        self.history["mean_fitness"].append(float(np.mean([p.fitness for p in self.particles])))
        
    def optimize(self, callback=None):
        # Run the PSO algorithm for the specified number of iterations.
        self.initialise()
        self.evaluate_particles()
        self.update_personal_bests()
        self.update_global_best()
        self._record()
        if callback:
            callback(0, self)
        stale = 0  # iterations in a row with no improvement (for early stopping)
        for _ in range(self.num_iterations):
            previous = self.global_best_fitness  # remember, to detect improvement
            self.assign_informants()
            # Move every particle: update velocity first, then position.
            for p in self.particles:
                inf_best_pos, _ = p.get_informants_best()
                p.update_velocity(
                    inf_best_pos, self.global_best_position,
                    self.alpha, self.beta, self.delta, self.gamma,
                )
                p.update_position(self.epsilon)
            self.evaluate_particles()
            self.update_personal_bests()
            self.update_global_best()
            self._record()
            if callback:
                callback(len(self.history["global_best"]) - 1, self)
                # Count consecutive iterations with no improvement for early stopping.
                stale = stale + 1 if self.global_best_fitness >= previous else 0
                if self.patience is not None and stale >= self.patience:
                    break
            return self.global_best_solution()
        
    def get_best_solution(self):
        # Return the best position and its fitness score.
        return self.global_best_position, self.global_best_fitness