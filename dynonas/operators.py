from __future__ import annotations

from typing import Callable, Optional
import numpy as np
from dynonas.population import Subnet, Population
from dynonas.encoding import random_alpha, random_b, N_EDGES, validate_individual


def tournament_select(individuals: list[Subnet], T: int,
                       rng: np.random.Generator) -> Subnet:

    candidates = rng.choice(len(individuals), size=min(T, len(individuals)),
                             replace=False)
    return max((individuals[i] for i in candidates), key=lambda x: x.fitness)


def crossover(p1: Subnet, p2: Subnet, t_c: float,
              rng: np.random.Generator) -> Subnet:

    def _mix(a1, b1, a2, b2):
        a_out = np.empty_like(a1)
        b_out = np.empty_like(b1)
        for row in range(N_EDGES):
            t1 = rng.random()
            if t1 > t_c:
                a_out[row] = a1[row]
                b_out[row] = b1[row]
            else:
                a_out[row] = a2[row]
                b_out[row] = b2[row]
        return a_out, b_out

    alpha_n, b_n = _mix(p1.alpha_normal, p1.b_normal,
                        p2.alpha_normal, p2.b_normal)
    alpha_r, b_r = _mix(p1.alpha_reduce, p1.b_reduce,
                        p2.alpha_reduce, p2.b_reduce)

    return Subnet(alpha_normal=alpha_n, b_normal=b_n,
                  alpha_reduce=alpha_r, b_reduce=b_r)


def mutate(individual: Subnet, t_m: float,
           rng: np.random.Generator) -> Subnet:

    def _mut(alpha, b):
        a_out = alpha.copy()
        b_out = b.copy()
        for row in range(N_EDGES):
            t2 = rng.random()
            if t2 < t_m:
                a_out[row] = rng.standard_normal(a_out.shape[1])
                new_b = np.zeros(b_out.shape[1], dtype=b_out.dtype)
                new_b[rng.integers(b_out.shape[1])] = 1
                b_out[row] = new_b
        return a_out, b_out

    alpha_n, b_n = _mut(individual.alpha_normal, individual.b_normal)
    alpha_r, b_r = _mut(individual.alpha_reduce, individual.b_reduce)
    return Subnet(alpha_normal=alpha_n, b_normal=b_n,
                  alpha_reduce=alpha_r, b_reduce=b_r)


def evolve(
    population: Population,
    t_c: float,
    t_m: float,
    T: int,
    rng: np.random.Generator,
    mutate_fn: Optional[Callable[[Subnet, np.random.Generator, Optional[Subnet]], Subnet]] = None,
) -> list[Subnet]:

    N = len(population)
    individuals = list(population.individuals)

    # (child, parent_reference) pairs — parent is None for the elite slot
    offspring_parents: list[tuple[Subnet, Optional[Subnet]]] = []

    if population.p_best is not None:
        offspring_parents.append((population.p_best.copy(), None))
    else:
        p = tournament_select(individuals, T, rng)
        offspring_parents.append((p.copy(), None))

    while len(offspring_parents) < N:
        p1 = tournament_select(individuals, T, rng)
        p2 = tournament_select(individuals, T, rng)
        child = crossover(p1, p2, t_c, rng)
        offspring_parents.append((child, p1))

    if mutate_fn is not None:
        mutated = [mutate_fn(child, rng, parent) for child, parent in offspring_parents]
    else:
        mutated = [mutate(child, t_m, rng) for child, _ in offspring_parents]

    # Reset fitness for newly created individuals
    for ind in mutated:
        ind.fitness = -1.0

    return mutated
