from __future__ import annotations
import sys
import os
import numpy as np
import torch
import torch.nn as nn

from dynonas.encoding import N_EDGES, N_OPS
from dynonas.population import Subnet, Population

_HERE = os.path.dirname(__file__)
_CNN = os.path.normpath(os.path.join(_HERE, '..', 'cnn'))
if _CNN not in sys.path:
    sys.path.insert(0, _CNN)


def _argmax_b(alpha_tensor: torch.Tensor) -> np.ndarray:
    """Convert [N_EDGES, N_OPS] tensor of arch params to one-hot b matrix."""
    b = np.zeros((N_EDGES, N_OPS), dtype=np.int8)
    indices = alpha_tensor.argmax(dim=1).cpu().numpy()
    b[np.arange(N_EDGES), indices] = 1
    return b


def local_search_step(
    subnet: Subnet,
    model: nn.Module,
    architect,
    train_iter,
    val_iter,
    lr: float,
    net_optimizer,
    unrolled: bool,
    device: torch.device,
) -> Subnet:

    result = subnet.copy()

    for cell_type in ('normal', 'reduce'):
        alpha = (result.alpha_normal if cell_type == 'normal'
                 else result.alpha_reduce)

        # Step 1: set all b=1 (couple all ops)
        b_full = np.ones((N_EDGES, N_OPS), dtype=np.int8)

        # Step 2: copy alpha into the supernet's arch params
        alpha_tensor = torch.tensor(alpha, dtype=torch.float32, device=device)
        with torch.no_grad():
            if cell_type == 'normal':
                model.alphas_normal.copy_(alpha_tensor)
            else:
                model.alphas_reduce.copy_(alpha_tensor)

    try:
        input_train, target_train = next(train_iter)
        input_val, target_val = next(val_iter)
    except StopIteration:
        return result   # exhausted; skip this individual

    input_train = input_train.to(device, non_blocking=True)
    target_train = target_train.to(device, non_blocking=True)
    input_val = input_val.to(device, non_blocking=True)
    target_val = target_val.to(device, non_blocking=True)

    architect.step(input_train, target_train, input_val, target_val,
                   lr, net_optimizer, unrolled=unrolled)

    with torch.no_grad():
        result.alpha_normal = model.alphas_normal.detach().cpu().numpy().copy()
        result.alpha_reduce = model.alphas_reduce.detach().cpu().numpy().copy()
        result.b_normal = _argmax_b(model.alphas_normal)
        result.b_reduce = _argmax_b(model.alphas_reduce)

    result.fitness = -1.0   # fitness is stale after modification
    return result


def local_search_population(
    population: Population,
    model: nn.Module,
    architect,
    train_loader,
    val_loader,
    lr: float,
    net_optimizer,
    unrolled: bool,
    device: torch.device,
) -> None:

    orig_normal = model.alphas_normal.data.clone()
    orig_reduce = model.alphas_reduce.data.clone()

    train_iter = iter(train_loader)
    val_iter = iter(val_loader)

    for i, ind in enumerate(population.individuals):
        population.individuals[i] = local_search_step(
            ind, model, architect,
            train_iter, val_iter,
            lr, net_optimizer, unrolled, device,
        )

    # Restore arch params after local search
    with torch.no_grad():
        model.alphas_normal.copy_(orig_normal)
        model.alphas_reduce.copy_(orig_reduce)
