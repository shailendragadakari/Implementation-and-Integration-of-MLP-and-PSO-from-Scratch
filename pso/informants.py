# Informant selection strategies.
# Both strategies exclude the particle itself and return exactly k particles.
# "Informants" are the neighbors a particle listens to; swapping the strategy
# changes who the neighbors are without touching any other code.

from abc import ABC, abstractmethod
import numpy as np

class InformantSelectionStrategy(ABC):
    @abstractmethod
    def select_informants(self, particle, population, k: int) -> list:
        # Return k other particles from `population` (a list of particles).
        pass

    @staticmethod
    def _check(k: int, n: int) -> None:
        # A particle needs at least 1 informant and at most everyone but itself.
        if not 1 <= k <= n - 1:
            raise ValueError(f"Need 1 <= k <= N-1 (k={k}, N={n})")
        
class RandomInformantSelection(InformantSelectionStrategy):
    # Pick k other particles completely at random.
    def __init__(self, seed=None):
        self._rng = np.random.default_rng(seed)

    def select_informants(self, particle, population, k):
        self._check(k, len(population))
        others = [p for p in population if p is not particle]  # leave out itself
        # replace=False -> no particle is chosen twice.
        idx = self._rng.choice(len(others), size=k, replace=False)
        return [others[i] for i in idx]
    
class NearestNeighbourInformantSelection(InformantSelectionStrategy):
    # k nearest particles by Euclidean distance between current positions.
    def select_informants(self, particle, population, k):
        self._check(k, len(population))
        others = [p for p in population if p is not particle]
        positions = np.stack([p.position for p in others])  # table: one row per other particle
        # Euclidean distance from this particle to each other one.
        dists = np.sqrt(np.sum((positions - particle.position) ** 2, axis=1))
        # argsort gives indices from closest to farthest; keep the first k.
        # "stable" makes ties resolve the same way every run.
        nearest = np.argsort(dists, kind="stable")[:k]
        return [others[i] for i in nearest]