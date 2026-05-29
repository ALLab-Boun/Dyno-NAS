from __future__ import annotations

import os
import sys
import time
import glob
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn

# Ensure cnn/ is importable
_HERE = os.path.dirname(__file__)
_ROOT = os.path.normpath(os.path.join(_HERE, '..'))
_CNN = os.path.join(_ROOT, 'cnn')
for _p in [_ROOT, _CNN]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

import utils                                  # cnn/utils.py
from model_search import Network              # cnn/model_search.py
from architect import Architect               # cnn/architect.py
import torch.utils.data
import torchvision.datasets as dset

from dynonas.config import get_args
from dynonas.population import Population, init_population
from dynonas.operators import evolve
from dynonas.evaluate import evaluate_population
from dynonas.local_search import local_search_population
from dynonas.encoding import table_to_genotype
from dynonas.mutations import (
    SoftMutationController, mutate_soft,
    AdaptiveMutationController, mutate_adaptive,
    ZeroMutationController, mutate_zero,
    PeriodicMutationController, mutate_guided,
    calculate_population_diversity, get_population_embeddings,
)


def _setup_logging(save_dir: str) -> None:
    utils.create_exp_dir(save_dir, scripts_to_save=glob.glob('*.py') +
                         glob.glob('dynonas/*.py'))
    log_format = '%(asctime)s %(message)s'
    logging.basicConfig(
        stream=sys.stdout, level=logging.INFO,
        format=log_format, datefmt='%m/%d %I:%M:%S %p')
    fh = logging.FileHandler(os.path.join(save_dir, 'log.txt'))
    fh.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(fh)


def _train_supernet_epoch(
    train_queue, val_queue, model, architect, criterion,
    optimizer, lr, args, device
):
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    model.train()

    for step, (images, labels) in enumerate(train_queue):
        if args.debug and step >= 2:
            break

        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        # Architecture parameter update
        try:
            img_val, lbl_val = next(_train_supernet_epoch._val_iter)
        except (StopIteration, AttributeError):
            _train_supernet_epoch._val_iter = iter(val_queue)
            img_val, lbl_val = next(_train_supernet_epoch._val_iter)

        img_val = img_val.to(device, non_blocking=True)
        lbl_val = lbl_val.to(device, non_blocking=True)

        architect.step(images, labels, img_val, lbl_val,
                       lr, optimizer, unrolled=args.unrolled)

        # Supernet weight update
        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        prec1, _ = utils.accuracy(logits, labels, topk=(1, 5))
        objs.update(loss.item(), images.size(0))
        top1.update(prec1.item(), images.size(0))

    return top1.avg, objs.avg


def _build_mutate_fn(
    mutation_type: str,
    t: int,
    G_max: int,
    soft_ctrl,
    adaptive_ctrl,
    zero_ctrl,
    periodic_ctrl,
    population,
    t_m: float,
    rng: np.random.Generator,
):

    if mutation_type == 'default' or mutation_type is None:
        return None

    if mutation_type == 'soft':
        def fn(ind, rng, parent):
            result, cr, f = mutate_soft(ind, soft_ctrl, rng)
            result._soft_cr = cr
            result._soft_f = f
            result._parent_fitness = parent.fitness if parent is not None else -1.0
            return result
        return fn

    if mutation_type == 'adaptive':
        def fn(ind, rng, parent):
            result, op_idx = mutate_adaptive(ind, adaptive_ctrl, rng)
            result._mutation_op_idx = op_idx
            result._parent_fitness = parent.fitness if parent is not None else -1.0
            return result
        return fn

    if mutation_type == 'zero':
        pop_fitnesses = [ind.fitness for ind in population.individuals if ind.fitness >= 0]
        pop_mean = float(np.mean(pop_fitnesses)) if pop_fitnesses else 0.0

        def fn(ind, rng, parent):
            p_fit = parent.fitness if parent is not None else -1.0
            result, was_skipped = mutate_zero(
                ind, zero_ctrl, rng,
                parent_fitness=p_fit,
                population_mean=pop_mean,
                fallback_t_m=t_m,
            )
            result._zero_skipped = was_skipped
            result._parent_fitness = p_fit
            return result
        return fn

    if mutation_type == 'hybrid':
        progress = t / max(G_max, 1)
        diversity = calculate_population_diversity(population.individuals)
        mutation_rate = periodic_ctrl.get_mutation_rate(diversity)
        pop_embeddings = get_population_embeddings(population.individuals)

        def fn(ind, rng, parent):
            if rng.random() > mutation_rate:
                # periodic rate acts as gate; below threshold → no mutation
                result = ind.copy()
                result.fitness = -1.0
                return result

            if progress < 0.3:
                result = mutate_guided(ind, pop_embeddings, rng, num_candidates=5)
                result._mutation_type = 'guided'
            elif progress < 0.7:
                result, op_idx = mutate_adaptive(ind, adaptive_ctrl, rng)
                result._mutation_op_idx = op_idx
                result._mutation_type = 'adaptive'
                result._parent_fitness = parent.fitness if parent is not None else -1.0
            else:
                result, cr, f = mutate_soft(ind, soft_ctrl, rng)
                result._soft_cr = cr
                result._soft_f = f
                result._parent_fitness = parent.fitness if parent is not None else -1.0
                result._mutation_type = 'soft'

            return result
        return fn

    return None


def _update_controllers(
    mutation_type: str,
    population,
    soft_ctrl,
    adaptive_ctrl,
    zero_ctrl,
    periodic_ctrl,
):
    """Update mutation controller state using the just-evaluated population."""
    if mutation_type == 'default':
        return

    pop_fitnesses = [ind.fitness for ind in population.individuals if ind.fitness >= 0]
    pop_mean = float(np.mean(pop_fitnesses)) if pop_fitnesses else 0.0

    for ind in population.individuals:
        if ind.fitness < 0:
            continue

        if mutation_type == 'soft' and hasattr(ind, '_soft_cr'):
            improved = ind.fitness > getattr(ind, '_parent_fitness', 0.0)
            soft_ctrl.update_memory(ind._soft_cr, ind._soft_f, improved)

        elif mutation_type == 'adaptive' and hasattr(ind, '_mutation_op_idx'):
            improved = ind.fitness > pop_mean
            adaptive_ctrl.update_credit(ind._mutation_op_idx, improved)

        elif mutation_type == 'zero' and hasattr(ind, '_zero_skipped'):
            zero_ctrl.update_statistics(ind._zero_skipped, ind.fitness)

        elif mutation_type == 'hybrid':
            m_type = getattr(ind, '_mutation_type', None)
            if m_type == 'adaptive' and hasattr(ind, '_mutation_op_idx'):
                improved = ind.fitness > pop_mean
                adaptive_ctrl.update_credit(ind._mutation_op_idx, improved)
            elif m_type == 'soft' and hasattr(ind, '_soft_cr'):
                improved = ind.fitness > getattr(ind, '_parent_fitness', 0.0)
                soft_ctrl.update_memory(ind._soft_cr, ind._soft_f, improved)

    if mutation_type == 'hybrid' and periodic_ctrl is not None:
        periodic_ctrl.update_generation()

    if mutation_type == 'zero' and zero_ctrl is not None:
        stats = zero_ctrl.get_statistics()
        logging.debug('zero mutation stats: %s', stats)


def main():
    args = get_args()

    args.save = '{}-{}'.format(args.save,
                               time.strftime('%Y%m%d-%H%M%S'))
    _setup_logging(args.save)
    logging.info('args = %s', args)

    # Reproducibility
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    if args.debug or not torch.cuda.is_available():
        device = torch.device('cpu')
        logging.info('Running on CPU (debug=%s)', args.debug)
    else:
        torch.cuda.set_device(args.gpu)
        cudnn.benchmark = True
        torch.cuda.manual_seed(args.seed)
        device = torch.device('cuda', args.gpu)
        logging.info('GPU device = %d', args.gpu)

    # ---- Data ---------------------------------------------------------------
    if args.dataset == 'cifar100':
        num_classes = 100
        train_transform, valid_transform = utils._data_transforms_cifar10(args)
        train_data = dset.CIFAR100(root=args.data, train=True,
                                   download=True, transform=train_transform)
    else:
        num_classes = 10
        train_transform, valid_transform = utils._data_transforms_cifar10(args)
        train_data = dset.CIFAR10(root=args.data, train=True,
                                  download=True, transform=train_transform)

    n_train = len(train_data)
    indices = list(range(n_train))
    split = int(np.floor(args.train_portion * n_train))

    train_queue = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size,
        sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[:split]),
        pin_memory=False, num_workers=args.num_workers)
    val_queue = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size,
        sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[split:]),
        pin_memory=False, num_workers=args.num_workers)

    # ---- Supernet -----------------------------------------------------------
    # cnn/model_search.py's _initialize_alphas hard-codes .cuda(); patch it for CPU.
    if device.type == 'cpu':
        from torch.autograd import Variable
        from genotypes import PRIMITIVES as _PRIM

        def _cpu_init_alphas(self):
            k = sum(1 for i in range(self._steps) for _ in range(2 + i))
            num_ops = len(_PRIM)
            self.alphas_normal = Variable(
                1e-3 * torch.randn(k, num_ops), requires_grad=True)
            self.alphas_reduce = Variable(
                1e-3 * torch.randn(k, num_ops), requires_grad=True)
            self._arch_parameters = [self.alphas_normal, self.alphas_reduce]

        Network._initialize_alphas = _cpu_init_alphas

    criterion = nn.CrossEntropyLoss().to(device)
    model = Network(args.init_channels, num_classes, args.layers, criterion)
    model = model.to(device)
    logging.info('Supernet params: %.2f MB',
                 utils.count_parameters_in_MB(model))

    optimizer = torch.optim.SGD(
        model.parameters(), lr=args.lr,
        momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, float(args.G_max), eta_min=args.lr_min)

    # Wrap args for Architect (needs .momentum, .weight_decay, .arch_learning_rate,
    # .arch_weight_decay attributes)
    class _ArchArgs:
        momentum = args.momentum
        weight_decay = args.weight_decay
        arch_learning_rate = args.arch_lr
        arch_weight_decay = args.arch_wd

    architect = Architect(model, _ArchArgs())

    # ---- Mutation controllers -----------------------------------------------
    soft_ctrl: SoftMutationController | None = None
    adaptive_ctrl: AdaptiveMutationController | None = None
    zero_ctrl: ZeroMutationController | None = None
    periodic_ctrl: PeriodicMutationController | None = None

    if args.mutation_type == 'soft':
        soft_ctrl = SoftMutationController()
        logging.info('Mutation type: soft')
    elif args.mutation_type == 'adaptive':
        adaptive_ctrl = AdaptiveMutationController()
        logging.info('Mutation type: adaptive')
    elif args.mutation_type == 'zero':
        zero_ctrl = ZeroMutationController(zero_mutation_prob=0.1, adaptive=True)
        logging.info('Mutation type: zero')
    elif args.mutation_type == 'hybrid':
        adaptive_ctrl = AdaptiveMutationController()
        soft_ctrl = SoftMutationController()
        periodic_ctrl = PeriodicMutationController(
            base_rate=args.mutation_prob, period=5, min_rate=0.05, max_rate=0.3)
        logging.info('Mutation type: hybrid')
    else:
        logging.info('Mutation type: default')

    # ---- Population ---------------------------------------------------------
    n_eval_batches = 2 if args.debug else None   # None = full val set
    population = init_population(args.pop_size, rng)
    logging.info('Initialized population of %d individuals', args.pop_size)

    evaluate_population(population, model, val_queue, criterion,
                        device, n_batches=n_eval_batches)
    population.update_best()
    logging.info('Initial best fitness: %.2f%%', population.p_best.fitness
                 if population.p_best else -1)

    for t in range(args.G_max):
        scheduler.step()
        lr = scheduler.get_last_lr()[0]

        train_acc, train_loss = _train_supernet_epoch(
            train_queue, val_queue, model, architect,
            criterion, optimizer, lr, args, device)
        logging.info('gen %03d | supernet train_acc=%.2f loss=%.4f lr=%.4e',
                     t, train_acc, train_loss, lr)

        if not args.disable_local_search and t % args.gradient_interval == 0:
            logging.info('gen %03d | running local search ...', t)
            local_search_population(
                population, model, architect,
                train_queue, val_queue,
                lr, optimizer, args.unrolled, device)

        if args.disable_crossover and args.disable_mutation:
            pass 
        else:
            t_m_eff = args.mutation_prob if not args.disable_mutation else 1.0
            t_c_eff = args.crossover_prob if not args.disable_crossover else 0.0

            mutate_fn = _build_mutate_fn(
                args.mutation_type, t, args.G_max,
                soft_ctrl, adaptive_ctrl, zero_ctrl, periodic_ctrl,
                population, t_m_eff, rng,
            )

            new_individuals = evolve(population,
                                     t_c=t_c_eff,
                                     t_m=t_m_eff,
                                     T=args.tournament_size,
                                     rng=rng,
                                     mutate_fn=mutate_fn)
            population.individuals = new_individuals

        evaluate_population(population, model, val_queue, criterion,
                            device, n_batches=n_eval_batches)

        population.update_best()
        population.ensure_elite()

        _update_controllers(
            args.mutation_type, population,
            soft_ctrl, adaptive_ctrl, zero_ctrl, periodic_ctrl,
        )

        if t % args.report_freq == 0 or t == args.G_max - 1:
            best = population.p_best
            logging.info('gen %03d | best fitness=%.2f%%', t,
                         best.fitness if best else -1)
            if best is not None:
                geno = table_to_genotype(
                    best.alpha_normal, best.b_normal,
                    best.alpha_reduce, best.b_reduce)
                logging.info('best genotype = %s', geno)

        utils.save(model, os.path.join(args.save, 'weights.pt'))

    best = population.p_best
    if best is None:
        logging.error('No best individual found — population never evaluated?')
        return

    geno = table_to_genotype(
        best.alpha_normal, best.b_normal,
        best.alpha_reduce, best.b_reduce)
    logging.info('Search finished.  Best fitness: %.2f%%', best.fitness)
    logging.info('Best genotype: %s', geno)

    geno_path = os.path.join(args.save, 'best_genotype.txt')
    with open(geno_path, 'w') as f:
        f.write(repr(geno) + '\n')
    logging.info('Genotype saved to %s', geno_path)


if __name__ == '__main__':
    main()
