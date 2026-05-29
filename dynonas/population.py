from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Optional

from dynonas.encoding import random_alpha, random_b, N_EDGES, N_OPS


@dataclass
class Subnet:
    alpha_normal: np.ndarray          # [N_EDGES, N_OPS]
    b_normal: np.ndarray              # [N_EDGES, N_OPS], one-hot per row
    alpha_reduce: np.ndarray          # [N_EDGES, N_OPS]
    b_reduce: np.ndarray              # [N_EDGES, N_OPS], one-hot per row
    fitness: float = -1.0             # val accuracy; -1 = unevaluated

    def copy(self) -> 'Subnet':
        return Subnet(
            alpha_normal=self.alpha_normal.copy(),
            b_normal=self.b_normal.copy(),
            alpha_reduce=self.alpha_reduce.copy(),
            b_reduce=self.b_reduce.copy(),
            fitness=self.fitness,
        )


def random_subnet(rng: Optional[np.random.Generator] = None) -> Subnet:
    """Sample a random valid Subnet: Gaussian α, random one-hot b."""
    if rng is None:
        rng = np.random.default_rng()
    return Subnet(
        alpha_normal=random_alpha(rng),
        b_normal=random_b(rng),
        alpha_reduce=random_alpha(rng),
        b_reduce=random_b(rng),
    )


class Population:
    """Container for a list of Subnet individuals with elitism tracking."""

    def __init__(self, individuals: list[Subnet]):
        self.individuals = individuals
        self.p_best: Optional[Subnet] = None
        self.update_best()

    def update_best(self):
        """Update p_best to the highest-fitness individual seen so far."""
        evaluated = [ind for ind in self.individuals if ind.fitness >= 0]
        if not evaluated:
            return
        candidate = max(evaluated, key=lambda x: x.fitness)
        if self.p_best is None or candidate.fitness > self.p_best.fitness:
            self.p_best = candidate.copy()

    def ensure_elite(self):
        """Re-insert p_best if it is not already present (by identity)."""
        if self.p_best is None:
            return
        fitnesses = [ind.fitness for ind in self.individuals]
        if self.p_best.fitness not in fitnesses:
            # Replace worst individual with p_best
            worst_idx = int(np.argmin([ind.fitness for ind in self.individuals]))
            self.individuals[worst_idx] = self.p_best.copy()

    def __len__(self) -> int:
        return len(self.individuals)

    def __iter__(self):
        return iter(self.individuals)


def init_population(N: int, rng: Optional[np.random.Generator] = None) -> Population:
    if rng is None:
        rng = np.random.default_rng()
    return Population([random_subnet(rng) for _ in range(N)])
