from __future__ import annotations

import numpy as np
import sys
import os
from typing import Optional


N_EDGES = 14   # (M^2 + 3M) / 2 for M=4
N_OPS = 8      # must match len(PRIMITIVES) in cnn/genotypes.py

EDGE_ORDER: list[tuple[int, int]] = [
    (0, 2), (1, 2),                          # rows 0-1:  edges into node 2
    (0, 3), (1, 3), (2, 3),                  # rows 2-4:  edges into node 3
    (0, 4), (1, 4), (2, 4), (3, 4),          # rows 5-8:  edges into node 4
    (0, 5), (1, 5), (2, 5), (3, 5), (4, 5),  # rows 9-13: edges into node 5
]
assert len(EDGE_ORDER) == N_EDGES

NODE_GROUPS: dict[int, list[int]] = {
    2: [0, 1],
    3: [2, 3, 4],
    4: [5, 6, 7, 8],
    5: [9, 10, 11, 12, 13],
}

OP_NAMES = [
    'none',
    'max_pool_3x3',
    'avg_pool_3x3',
    'skip_connect',
    'sep_conv_3x3',
    'sep_conv_5x5',
    'dil_conv_3x3',
    'dil_conv_5x5',
]
assert len(OP_NAMES) == N_OPS
OP_INDEX = {name: i for i, name in enumerate(OP_NAMES)}


def random_alpha(rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """Return a [N_EDGES, N_OPS] float64 matrix of raw (unnormalized) weights."""
    if rng is None:
        rng = np.random.default_rng()
    return rng.standard_normal((N_EDGES, N_OPS))


def random_b(rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """Return a [N_EDGES, N_OPS] int array; exactly one 1 per row (random one-hot)."""
    if rng is None:
        rng = np.random.default_rng()
    b = np.zeros((N_EDGES, N_OPS), dtype=np.int8)
    chosen = rng.integers(0, N_OPS, size=N_EDGES)
    b[np.arange(N_EDGES), chosen] = 1
    return b


def softmax_alpha(alpha: np.ndarray) -> np.ndarray:
    """Row-wise softmax of alpha [N_EDGES, N_OPS] → normalized weights in (0,1)."""
    shifted = alpha - alpha.max(axis=1, keepdims=True)
    exp_a = np.exp(shifted)
    return exp_a / exp_a.sum(axis=1, keepdims=True)


def prune_top2(alpha: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:

    alpha_norm = softmax_alpha(alpha)                   # [14, 8]
    # Best alpha value per edge = max over ops of (alpha_norm * b)
    edge_scores = (alpha_norm * b).max(axis=1)          # [14]

    alpha_p = np.zeros_like(alpha)
    b_p = np.zeros_like(b)

    for node, rows in NODE_GROUPS.items():
        scores = edge_scores[rows]
        # indices of the top-2 rows (relative to the rows slice)
        top2_rel = np.argsort(scores)[-2:]
        top2_abs = [rows[r] for r in top2_rel]
        for r in top2_abs:
            alpha_p[r] = alpha[r]
            b_p[r] = b[r]

    return alpha_p, b_p


def _add_cnn_to_path():
    here = os.path.dirname(__file__)
    cnn_path = os.path.join(here, '..', 'cnn')
    cnn_path = os.path.normpath(cnn_path)
    if cnn_path not in sys.path:
        sys.path.insert(0, cnn_path)


def table_to_genotype(
    alpha_normal: np.ndarray,
    b_normal: np.ndarray,
    alpha_reduce: np.ndarray,
    b_reduce: np.ndarray,
):

    _add_cnn_to_path()
    from genotypes import Genotype  # noqa: imported from cnn/

    def _cell_gene(alpha, b):
        _, b_p = prune_top2(alpha, b)
        gene = []
        for node in sorted(NODE_GROUPS.keys()):
            rows = NODE_GROUPS[node]
            active = [(r, int(np.argmax(b_p[r]))) for r in rows if b_p[r].sum() > 0]
            # Sort by edge index so genotype order is deterministic
            active.sort(key=lambda x: x[0])
            for row_idx, op_idx in active:
                src, _ = EDGE_ORDER[row_idx]
                gene.append((OP_NAMES[op_idx], src))
        return gene

    concat = list(range(2, 2 + len(NODE_GROUPS)))  # [2,3,4,5]
    return Genotype(
        normal=_cell_gene(alpha_normal, b_normal),
        normal_concat=concat,
        reduce=_cell_gene(alpha_reduce, b_reduce),
        reduce_concat=concat,
    )


def genotype_to_table(genotype) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

    def _fill(gene_list):
        alpha = np.zeros((N_EDGES, N_OPS), dtype=np.float64)
        b = np.zeros((N_EDGES, N_OPS), dtype=np.int8)

        node_order = sorted(NODE_GROUPS.keys())
        idx = 0
        for node in node_order:
            rows = NODE_GROUPS[node]
            # Up to 2 active edges per node
            for _ in range(2):
                if idx >= len(gene_list):
                    break
                op_name, src = gene_list[idx]
                idx += 1
                dst = node
                edge = (src, dst)
                if edge in EDGE_ORDER and op_name in OP_INDEX:
                    row = EDGE_ORDER.index(edge)
                    op_i = OP_INDEX[op_name]
                    alpha[row, op_i] = 1.0
                    b[row, op_i] = 1
        return alpha, b

    alpha_n, b_n = _fill(genotype.normal)
    alpha_r, b_r = _fill(genotype.reduce)
    return alpha_n, b_n, alpha_r, b_r


def alpha_D_vector(alpha: np.ndarray) -> np.ndarray:
    """Flatten [N_EDGES, N_OPS] → [N_EDGES * N_OPS] = [112] for one cell."""
    assert alpha.shape == (N_EDGES, N_OPS), f"Expected ({N_EDGES},{N_OPS}), got {alpha.shape}"
    return alpha.flatten()


def validate_individual(alpha: np.ndarray, b: np.ndarray) -> bool:
    """Return True iff b has exactly one 1 per row (one-hot invariant)."""
    if alpha.shape != (N_EDGES, N_OPS) or b.shape != (N_EDGES, N_OPS):
        return False
    row_sums = b.sum(axis=1)
    return bool(np.all(row_sums == 1))
