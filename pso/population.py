# Population class. Owns the particles, global best and the main loop.
#
# "Alg. 39 line N" comments refer to Algorithm 39 (PSO with informants) in Luke,
# "Essentials of Metaheuristics", Section 3.5, as numbered in the Week 7 lecture slides.
# NOTE: double-check these line numbers against your copy of the book / slides.
#
# Extensions beyond the minimum specification (all off by default, so the default
# behaviour is exactly Algorithm 39):
#   * boundary handling: "none", "clamp", "reflect" or "random" (see _apply_boundary)
#   * inertia schedule : "constant" or "linear" (alpha decreases from `alpha` to `inertia_end`)
# The Greek-letter names follow the specification: alpha = inertia, beta = cognitive pull,
# delta = informant pull, gamma = global best pull, epsilon = step size.
import numpy as np

from pso.fitness import FitnessFunction
from pso.informants import InformantSelectionStrategy
from pso.particle import Particle


BOUNDARY_MODES = ("none", "clamp", "reflect", "random")
INERTIA_SCHEDULES = ("constant", "linear")


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
        boundary: str = "none",              # extension: what to do when a particle leaves [-bound, bound]
        bound: float = 5.0,                  # half-width of the allowed box (used if boundary != "none")
        inertia_schedule: str = "constant",  # extension: "constant" or "linear"
        inertia_end: float = 0.4,            # final alpha when inertia_schedule == "linear"
    ):
        # Validate early so mistakes give a clear message instead of a confusing crash later.
        if not 1 <= num_informants <= num_particles - 1:
            raise ValueError("Need 1 <= num_informants <= num_particles - 1")
        if boundary not in BOUNDARY_MODES:
            raise ValueError(f"boundary must be one of {BOUNDARY_MODES}")
        if inertia_schedule not in INERTIA_SCHEDULES:
            raise ValueError(f"inertia_schedule must be one of {INERTIA_SCHEDULES}")
        if bound <= 0:
            raise ValueError("bound must be positive")
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
        self.boundary = boundary
        self.bound = bound
        self.inertia_schedule = inertia_schedule
        self.inertia_end = inertia_end
        self.patience = patience  # stop if global best unchanged this many iterations
        self._rng = np.random.default_rng(seed)  # seed -> repeatable runs

        self.particles: list[Particle] = []
        self.global_best_position = None   # best position found by ANY particle so far
        self.global_best_fitness = np.inf
        # Convergence record: index 0 = after initial evaluation, then one per iteration.
        self.history = {"global_best": [], "mean_fitness": []}
        self.n_evaluations = 0  # how many times the fitness function has been called

    # ---- pseudocode steps --------------------------------------------------
    def initialise(self) -> None:
        # Alg. 39 lines 7-10: P <- {}; add `swarmsize` random particles with random
        #         initial velocities; Best <- empty (the global best is set after evaluation).
        r = self.init_range
        # Scatter particles uniformly in [-r, r] in every dimension. Starting
        # velocities are small (10% of that range) so early moves are gentle.
        self.particles = [
            Particle(
                position=self._rng.uniform(-r, r, self.dimension),
                velocity=self._rng.uniform(-r, r, self.dimension) * 0.1,
                rng=np.random.default_rng(self._rng.integers(2**32)),  # each particle gets its own generator
            )
            for _ in range(self.num_particles)
        ]
        self.global_best_position = None
        self.global_best_fitness = np.inf
        self.history = {"global_best": [], "mean_fitness": []}
        self.n_evaluations = 0

    def evaluate_particles(self) -> None:
        # Alg. 39 lines 12-13: for each particle, AssessFitness.
        # Score every particle at its current position.
        for p in self.particles:
            p.evaluate(self.fitness_function)
        self.n_evaluations += len(self.particles)

    def update_personal_bests(self) -> None:
        for p in self.particles:
            p.update_personal_best()

    def update_global_best(self) -> None:
        # Alg. 39 lines 14-15: if Best is empty or x is fitter than Best then Best <- x.
        #         (Fitness is minimised here, so "fitter" means a smaller value.)
        # The swarm's best is the best personal best; only replace the record if it improved.
        best = min(self.particles, key=lambda p: p.best_fitness)
        if best.best_fitness < self.global_best_fitness:
            self.global_best_fitness = best.best_fitness
            self.global_best_position = best.best_position.copy()

    def assign_informants(self) -> None:
        # Alg. 39 line 18 (informants of x): choose who each particle listens to.
        # Re-pick neighbours every iteration (for nearest-neighbour this tracks movement).
        for p in self.particles:
            p.informants = self.informant_strategy.select_informants(
                p, self.particles, self.num_informants
            )

    def _alpha_at(self, t: int) -> float:
        # Inertia used in iteration t (0-based). Constant unless a schedule is chosen.
        if self.inertia_schedule == "constant" or self.num_iterations <= 1:
            return self.alpha
        frac = t / (self.num_iterations - 1)
        return self.alpha + (self.inertia_end - self.alpha) * frac  # linear alpha -> inertia_end

    def _apply_boundary(self, p: Particle) -> None:
        # Extension (not in Algorithm 39): keep each coordinate inside [-bound, bound].
        #   none   : do nothing (the plain algorithm; particles may leave the box)
        #   clamp  : put the coordinate on the wall and zero that velocity component
        #   reflect: bounce back inside by the overshoot, and reverse that velocity component
        #   random : re-draw the offending coordinate uniformly inside the box, zero its velocity
        if self.boundary == "none":
            return
        B = self.bound
        out = np.abs(p.position) > B
        if not out.any():
            return
        if self.boundary == "clamp":
            p.position = np.clip(p.position, -B, B)
            p.velocity[out] = 0.0
        elif self.boundary == "reflect":
            pos = p.position.copy()
            pos[pos > B] = 2 * B - pos[pos > B]
            pos[pos < -B] = -2 * B - pos[pos < -B]
            p.position = np.clip(pos, -B, B)  # clip in case of a very large overshoot
            p.velocity[out] *= -1.0
        else:  # "random"
            p.position[out] = self._rng.uniform(-B, B, int(out.sum()))
            p.velocity[out] = 0.0

    def _record(self) -> None:
        self.history["global_best"].append(self.global_best_fitness)
        self.history["mean_fitness"].append(float(np.mean([p.fitness for p in self.particles])))

    def optimise(self, callback=None):
        # Run PSO. `callback(iteration, population)` (optional) is called after the initial
        #         evaluation (iteration 0) and after every iteration, e.g. for live progress.
        # Alg. 39 line 11 (repeat ...): initialise, then assess the first generation.
        self.initialise()
        self.evaluate_particles()
        self.update_personal_bests()
        self.update_global_best()
        self._record()
        if callback:
            callback(0, self)
        stale = 0  # iterations in a row with no improvement (for early stopping)
        for t in range(self.num_iterations):  # Alg. 39 lines 11 and 25: repeat ... until out of time
            alpha_t = self._alpha_at(t)
            previous = self.global_best_fitness  # remember, to detect improvement
            self.assign_informants()
            # Move every particle: velocity first, then position.
            # Alg. 39 lines 16-22 (velocity update) and 23-24 (position update).
            # The bests used here were fixed at the end of the previous iteration, so moving
            # the particles one after another gives the same result as moving them together.
            for p in self.particles:
                inf_best_pos, _ = p.get_informants_best()
                p.update_velocity(
                    inf_best_pos, self.global_best_position,
                    alpha_t, self.beta, self.gamma, self.delta,
                )
                p.update_position(self.epsilon)
                self._apply_boundary(p)  # extension; does nothing when boundary == "none"
            self.evaluate_particles()          # Alg. 39 lines 12-13 (next generation)
            self.update_personal_bests()
            self.update_global_best()          # Alg. 39 lines 14-15
            self._record()
            if callback:
                callback(len(self.history["global_best"]) - 1, self)
            # Count consecutive iterations without improvement; stop early if patience is set.
            stale = stale + 1 if self.global_best_fitness >= previous else 0
            if self.patience is not None and stale >= self.patience:
                break
        return self.get_best_solution()

    def get_best_solution(self):
        # Alg. 39 line 26: return Best
        return self.global_best_position, self.global_best_fitness