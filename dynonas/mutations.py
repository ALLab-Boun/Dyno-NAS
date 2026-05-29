from __future__ import annotations

import numpy as np
from dynonas.population import Subnet
from dynonas.encoding import N_EDGES, N_OPS


class SoftMutationController:

    def __init__(self, memory_size: int = 10):
        self.memory_size = memory_size
        self.MCR: list[float] = [0.5]
        self.MF: list[float] = [0.5]
        self._scr: list[float] = []
        self._sf: list[float] = []

    def generate_parameters(self, rng: np.random.Generator) -> tuple[float, float]:
        idx = int(rng.integers(len(self.MCR)))
        cr = float(np.clip(rng.normal(self.MCR[idx], 0.1), 0.0, 1.0))
        f = float(np.clip(rng.standard_cauchy() * 0.1 + self.MF[idx], 0.0, 1.0))
        return cr, f

    def update_memory(self, cr: float, f: float, fitness_improved: bool) -> None:
        if fitness_improved:
            self._scr.append(cr)
            self._sf.append(f)
        if len(self._scr) >= 5:
            self.MCR.append(float(np.mean(self._scr)))
            self.MF.append(float(np.mean(self._sf)))
            if len(self.MCR) > self.memory_size:
                self.MCR.pop(0)
                self.MF.pop(0)
            self._scr.clear()
            self._sf.clear()


def mutate_soft(ind: Subnet, controller: SoftMutationController,
                rng: np.random.Generator) -> tuple[Subnet, float, float]:

    cr, f = controller.generate_parameters(rng)
    result = ind.copy()

    for alpha, b in [
        (result.alpha_normal, result.b_normal),
        (result.alpha_reduce, result.b_reduce),
    ]:
        # Soft alpha perturbation
        for i in range(N_EDGES):
            if rng.random() < cr:
                alpha[i] += rng.standard_normal(N_OPS) * f * 0.01

        # Probabilistic op switching on b
        num_edges = max(1, int(N_EDGES * f))
        edges = rng.choice(N_EDGES, num_edges, replace=False)
        for edge_idx in edges:
            if rng.random() < cr:
                current_op = int(np.argmax(b[edge_idx]))
                probs = np.full(N_OPS, 1.0 / (N_OPS - 1))
                probs[current_op] = 0.0
                new_op = int(rng.choice(N_OPS, p=probs))
                b[edge_idx] = 0
                b[edge_idx, new_op] = 1

    result.fitness = -1.0
    return result, cr, f


class AdaptiveMutationController:

    def __init__(self, num_operations: int = N_OPS, memory_size: int = 10):
        self.num_operations = num_operations
        self.memory_size = memory_size
        # credit_memory[op, 0] = success count, [op, 1] = total count
        self.credit_memory = np.zeros((num_operations, 2))
        self.operation_probs = np.ones(num_operations) / num_operations

    def update_credit(self, op_idx: int, fitness_improved: bool) -> None:
        self.credit_memory[op_idx, 1] += 1
        if fitness_improved:
            self.credit_memory[op_idx, 0] += 1
        # Laplace-smoothed success rates → softmax at temperature 0.5
        rates = (self.credit_memory[:, 0] + 1) / (self.credit_memory[:, 1] + 2)
        exp_r = np.exp(rates / 0.5)
        self.operation_probs = exp_r / exp_r.sum()

    def get_operation_probabilities(self) -> np.ndarray:
        return self.operation_probs.copy()


def mutate_adaptive(ind: Subnet, controller: AdaptiveMutationController,
                    rng: np.random.Generator) -> tuple[Subnet, int]:

    result = ind.copy()
    op_probs = controller.get_operation_probabilities()

    if rng.random() < 0.5:
        alpha, b = result.alpha_normal, result.b_normal
    else:
        alpha, b = result.alpha_reduce, result.b_reduce

    row_idx = int(rng.integers(N_EDGES))
    mutated_op = int(rng.choice(N_OPS, p=op_probs))

    alpha[row_idx] = rng.standard_normal(N_OPS)
    b[row_idx] = 0
    b[row_idx, mutated_op] = 1

    result.fitness = -1.0
    return result, mutated_op


class ZeroMutationController:

    def __init__(self, zero_mutation_prob: float = 0.1, adaptive: bool = True):
        self.base_prob = zero_mutation_prob
        self.adaptive = adaptive
        self.current_prob = zero_mutation_prob
        self._zero_fit: list[float] = []
        self._active_fit: list[float] = []
        self.zero_count = 0
        self.active_count = 0

    def should_skip_mutation(self, parent_fitness: float,
                              population_mean: float) -> bool:
        if self.adaptive and parent_fitness >= 0.0:
            if parent_fitness > population_mean:
                prob = min(self.base_prob * 1.5, 0.5)
            else:
                prob = self.base_prob * 0.5
        else:
            prob = self.base_prob
        self.current_prob = prob
        return bool(np.random.rand() < prob)

    def update_statistics(self, was_skipped: bool, fitness: float) -> None:
        if was_skipped:
            self.zero_count += 1
            self._zero_fit.append(fitness)
        else:
            self.active_count += 1
            self._active_fit.append(fitness)

    def get_statistics(self) -> dict:
        return {
            'zero_count': self.zero_count,
            'active_count': self.active_count,
            'zero_mean_fitness': float(np.mean(self._zero_fit)) if self._zero_fit else 0.0,
            'active_mean_fitness': float(np.mean(self._active_fit)) if self._active_fit else 0.0,
        }


def mutate_zero(ind: Subnet, controller: ZeroMutationController,
                rng: np.random.Generator,
                parent_fitness: float = -1.0,
                population_mean: float = 0.0,
                fallback_t_m: float = 0.1) -> tuple[Subnet, bool]:

    from dynonas.operators import mutate as _default_mutate

    skip = controller.should_skip_mutation(parent_fitness, population_mean)
    if skip:
        result = ind.copy()
        result.fitness = -1.0
        return result, True
    else:
        return _default_mutate(ind, fallback_t_m, rng), False


def _embedding(ind: Subnet) -> np.ndarray:
    """Flatten b_normal + b_reduce into a 1-D binary vector."""
    return np.concatenate([ind.b_normal.flatten(), ind.b_reduce.flatten()])


def calculate_population_diversity(population) -> float:
    """Normalised mean pairwise Hamming distance over b-matrices."""
    inds = list(population)
    if len(inds) < 2:
        return 0.0
    embs = [_embedding(ind) for ind in inds]
    max_diff = embs[0].size
    diffs = [
        np.sum(embs[i] != embs[j])
        for i in range(len(embs))
        for j in range(i + 1, len(embs))
    ]
    return float(np.mean(diffs) / max_diff)


def get_population_embeddings(population) -> list[np.ndarray]:
    return [_embedding(ind) for ind in population]


def mutate_guided(ind: Subnet, population_embeddings: list[np.ndarray] | None,
                  rng: np.random.Generator,
                  num_candidates: int = 5) -> Subnet:

    candidates: list[Subnet] = []
    scores: list[float] = []

    for _ in range(num_candidates):
        cand = ind.copy()

        if rng.random() < 0.5:
            alpha, b = cand.alpha_normal, cand.b_normal
        else:
            alpha, b = cand.alpha_reduce, cand.b_reduce

        row_idx = int(rng.integers(N_EDGES))
        op_idx = int(rng.integers(N_OPS))
        alpha[row_idx] = rng.standard_normal(N_OPS)
        b[row_idx] = 0
        b[row_idx, op_idx] = 1
        cand.fitness = -1.0

        if population_embeddings:
            emb = _embedding(cand)
            diversity = float(np.mean([np.sum(emb != pe) for pe in population_embeddings]))
        else:
            diversity = float(rng.random())

        candidates.append(cand)
        scores.append(diversity)

    return candidates[int(np.argmax(scores))]


class PeriodicMutationController:

    def __init__(self, base_rate: float = 0.1, period: int = 5,
                 min_rate: float = 0.05, max_rate: float = 0.3):
        self.base_rate = base_rate
        self.period = period
        self.min_rate = min_rate
        self.max_rate = max_rate
        self.generation = 0
        self.current_rate = base_rate
        self._diversity_history: list[float] = []

    def get_mutation_rate(self, population_diversity: float | None = None) -> float:
        phase = (self.generation % self.period) / self.period
        rate = (self.base_rate
                + (self.max_rate - self.base_rate)
                * 0.5 * (1.0 + np.sin(2.0 * np.pi * phase)))

        if population_diversity is not None:
            self._diversity_history.append(population_diversity)
            if len(self._diversity_history) > 10:
                self._diversity_history.pop(0)
            if len(self._diversity_history) >= 5:
                recent = float(np.mean(self._diversity_history[-5:]))
                old = float(np.mean(self._diversity_history[:5]))
                if recent < old * 0.8:
                    rate = min(rate * 1.2, self.max_rate)

        self.current_rate = float(np.clip(rate, self.min_rate, self.max_rate))
        return self.current_rate

    def update_generation(self) -> None:
        self.generation += 1
