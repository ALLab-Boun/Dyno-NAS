from __future__ import annotations

import sys
import os
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from dynonas.encoding import prune_top2, softmax_alpha, N_EDGES, N_OPS
from dynonas.population import Subnet, Population

_HERE = os.path.dirname(__file__)
_CNN = os.path.normpath(os.path.join(_HERE, '..', 'cnn'))
if _CNN not in sys.path:
    sys.path.insert(0, _CNN)


def set_supernet_alphas(model: nn.Module, subnet: Subnet,
                        device: torch.device) -> None:

    with torch.no_grad():
        model.alphas_normal.copy_(
            torch.tensor(subnet.alpha_normal, dtype=torch.float32, device=device))
        model.alphas_reduce.copy_(
            torch.tensor(subnet.alpha_reduce, dtype=torch.float32, device=device))


def _pruned_weights(subnet: Subnet, device: torch.device):

    def _compute(alpha, b):
        alpha_p, b_p = prune_top2(alpha, b)
        w = softmax_alpha(alpha_p) * b_p          # zero out inactive edges
        return torch.tensor(w, dtype=torch.float32, device=device)

    w_n = _compute(subnet.alpha_normal, subnet.b_normal)
    w_r = _compute(subnet.alpha_reduce, subnet.b_reduce)
    return w_n, w_r


def eval_subnet_fitness(
    subnet: Subnet,
    model: nn.Module,
    val_loader,
    criterion: nn.Module,
    device: torch.device,
    n_batches: Optional[int] = None,
) -> float:

    import utils  # noqa: cnn/utils.py

    w_n, w_r = _pruned_weights(subnet, device)

    model.eval()
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()

    with torch.no_grad():
        for step, (images, labels) in enumerate(val_loader):
            if n_batches is not None and step >= n_batches:
                break
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            model.alphas_normal.data.copy_(torch.log(w_n.clamp(min=1e-8)))
            model.alphas_reduce.data.copy_(torch.log(w_r.clamp(min=1e-8)))

            logits = model(images)
            prec1, prec5 = utils.accuracy(logits, labels, topk=(1, 5))
            top1.update(prec1.item(), images.size(0))
            top5.update(prec5.item(), images.size(0))

    return top1.avg


def evaluate_population(
    population: Population,
    model: nn.Module,
    val_loader,
    criterion: nn.Module,
    device: torch.device,
    n_batches: Optional[int] = None,
) -> None:
    """Evaluate fitness for every individual in the population (in-place)."""
    # Save original arch params to restore after evaluation loop.
    orig_normal = model.alphas_normal.data.clone()
    orig_reduce = model.alphas_reduce.data.clone()

    for ind in population.individuals:
        ind.fitness = eval_subnet_fitness(
            ind, model, val_loader, criterion, device, n_batches)

    # Restore
    with torch.no_grad():
        model.alphas_normal.copy_(orig_normal)
        model.alphas_reduce.copy_(orig_reduce)

    population.update_best()
