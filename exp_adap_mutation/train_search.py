import os
import sys
import time
import glob
import numpy as np
import torch
import utils
import logging
import argparse
import torch.nn as nn
import torch.utils
import torch.nn.functional as F
import torchvision.datasets as dset
import torch.backends.cudnn as cudnn

from torch.autograd import Variable
from model_search import Network
from architect import Architect
from individual import SubnetIndividual
from evolution import tournament_selection, row_wise_crossover
from evolution_enhanced import (
    SoftMutationController, 
    mutate_soft,
    AdaptiveMutationController,
    mutate_adaptive,
    calculate_population_diversity
)
import copy


parser = argparse.ArgumentParser("cifar")
parser.add_argument('--data', type=str, default='/opt/app-root/src/thesiswork/data', help='location of the data corpus')
parser.add_argument('--batch_size', type=int, default=128, help='batch size')
parser.add_argument('--learning_rate', type=float, default=0.025, help='init learning rate')
parser.add_argument('--learning_rate_min', type=float, default=0.001, help='min learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
parser.add_argument('--report_freq', type=float, default=50, help='report frequency')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--epochs', type=int, default=50, help='num of training epochs')
parser.add_argument('--init_channels', type=int, default=16, help='num of init channels')
parser.add_argument('--layers', type=int, default=8, help='total number of layers')
parser.add_argument('--model_path', type=str, default='saved_models', help='path to save the model')
parser.add_argument('--cutout', action='store_true', default=True, help='use cutout')
parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
parser.add_argument('--drop_path_prob', type=float, default=0.3, help='drop path probability')
parser.add_argument('--save', type=str, default='EXP', help='experiment name')
parser.add_argument('--seed', type=int, default=2, help='random seed')
parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
parser.add_argument('--train_portion', type=float, default=0.5, help='portion of training data')
parser.add_argument('--unrolled', action='store_true', default=True, help='use one-step unrolled validation loss')
parser.add_argument('--arch_learning_rate', type=float, default=3e-4, help='learning rate for arch encoding')
parser.add_argument('--arch_weight_decay', type=float, default=1e-3, help='weight decay for arch encoding')
# new args
parser.add_argument('--pop_size', type=int, default=50, help='population size')
parser.add_argument('--g_max', type=int, default=50, help='maximum number of generations')
parser.add_argument('--tournament_size', type=int, default=20, help='tournament selection size')
parser.add_argument('--crossover_rate', type=float, default=0.5, help='crossover rate')
parser.add_argument('--mutation_rate', type=float, default=0.1, help='mutation rate')

args = parser.parse_args()

args.save = 'search-{}-{}'.format(args.save, time.strftime("%Y%m%d-%H%M%S"))
utils.create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)


CIFAR_CLASSES = 10


def main():
    if not torch.cuda.is_available():
        logging.info('no gpu device available')
        sys.exit(1)

    np.random.seed(args.seed)
    torch.cuda.set_device(args.gpu)
    cudnn.benchmark = True
    torch.manual_seed(args.seed)
    cudnn.enabled=True
    torch.cuda.manual_seed(args.seed)
    logging.info('gpu device = %d' % args.gpu)
    logging.info("args = %s", args)

    criterion = nn.CrossEntropyLoss()
    criterion = criterion.cuda()
    model = Network(args.init_channels, CIFAR_CLASSES, args.layers, criterion)
    model = model.cuda()
    logging.info("param size = %fMB", utils.count_parameters_in_MB(model))

    optimizer = torch.optim.SGD(
          model.parameters(),
          args.learning_rate,
          momentum=args.momentum,
          weight_decay=args.weight_decay)

    train_transform, valid_transform = utils._data_transforms_cifar10(args)
    train_data = dset.CIFAR10(root=args.data, train=True, download=False, transform=train_transform)

    num_train = len(train_data)
    indices = list(range(num_train))
    split = int(np.floor(args.train_portion * num_train))

    train_queue = torch.utils.data.DataLoader(
          train_data, batch_size=args.batch_size,
          sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[:split]),
          pin_memory=True, num_workers=2)

    valid_queue = torch.utils.data.DataLoader(
          train_data, batch_size=args.batch_size,
          sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[split:num_train]),
          pin_memory=True, num_workers=2)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, float(args.epochs), eta_min=args.learning_rate_min)

    architect = Architect(model, args)

    soft_controller = SoftMutationController(memory_size=10)
    adaptive_controller = AdaptiveMutationController(
        num_operations=8,  # 8 operations in GENAS (see genotypes.py PRIMITIVES)
        memory_size=10,
        target_size_ratio=1.2  # Allow 20% larger models
    )

    # Initialize a population of N subnet individuals.

    population = []
    logging.info(f"GENAS: Initializing a population with {args.pop_size} subnet individuals.")

    current_alphas_normal = model.alphas_normal
    current_alphas_reduce = model.alphas_reduce

    for i in range(args.pop_size):
        individual = SubnetIndividual(current_alphas_normal, current_alphas_reduce)
        individual.mask_normal = individual.mask_normal.cuda()
        individual.mask_reduce = individual.mask_reduce.cuda()
        individual.alphas_normal = individual.alphas_normal.cuda()
        individual.alphas_reduce = individual.alphas_reduce.cuda()
        population.append(individual)

    logging.info("GENAS: Initialization complete.")

    t = 0

    while t < args.g_max:

        logging.info(f"Generation {t} started.")

        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        logging.info(f"Current learning rate: {current_lr}")

        # Encode each subnet by means of the tabular encoding and train the supernet.

        model.train()
        train_loss = 0
        
        for step, (input, target) in enumerate(train_queue):
            input = input.cuda()
            target = target.cuda(non_blocking=True)
            optimizer.zero_grad()
            logits = model(input)
            loss = criterion(logits, target)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            train_loss += loss.item()
            
            if step % args.report_freq == 0:
                logging.info(f"Step: {step:03d} Loss: {loss.item():.4f}")

        # Represent each candidate architecture as a vector of operation weight parameters αD. Apply gradient descent to all the candidate architectures to locally optimize them.

        if t % 5 == 0:
            logging.info(f"Generation {t}: Performing Gradient-Guided Local Optimization on Population...")
            
            # Doğrulama setinden bir batch al (veya tüm valid set üzerinde dönebilirsin)
            input_valid, target_valid = next(iter(valid_queue))
            input_valid = input_valid.cuda()
            target_valid = target_valid.cuda(non_blocking=True)

            for ind_idx, individual in enumerate(population):
                # Her birey bir 'vektör' (alphas) olarak temsil edilir ve optimize edilir
                # Alphas parametreleri için gradyan takibini aç
                individual.alphas_normal.requires_grad = True
                individual.alphas_reduce.requires_grad = True
                
                # Yerel optimizasyon adımı
                architect.local_step(
                    input_valid, target_valid, 
                    individual, 
                    criterion,
                    lr=args.arch_learning_rate
                )
                
                # Hard Pruning
                individual.update_mask_from_alphas()

        
        # Evolve population P by applying evolutionary operators with soft mutation.

        logging.info(f"Generation {t}: Evolving population with soft mutation...")
        
        # Sort population and preserve elite
        population.sort(key=lambda x: x.fitness if x.fitness is not None else -1, reverse=True)
        elite_individual = copy.deepcopy(population[0])
        
        new_population = []
        while len(new_population) < args.pop_size:
            parent1 = tournament_selection(population, args.tournament_size)
            parent2 = tournament_selection(population, args.tournament_size)
            
            # Crossover
            if np.random.rand() < args.crossover_rate:
                child1, child2 = row_wise_crossover(parent1, parent2)
            else:
                child1, child2 = copy.deepcopy(parent1), copy.deepcopy(parent2)
            
            # Calculate progress for phase-based mutation
            progress = t / args.g_max
            
            # Hybrid Mutation Strategy
            for child, parent in [(child1, parent1), (child2, parent2)]:
                if np.random.rand() < args.mutation_rate:
                    if progress < 0.3:
                        # Early phase: Soft mutation for exploration
                        cr, f = mutate_soft(child, soft_controller)
                        child._soft_params = (cr, f)
                        child._parent_fitness = parent.fitness if parent.fitness is not None else 0.0
                    else:
                        # Late phase: Adaptive mutation for learning
                        mutated_op_idx = mutate_adaptive(child, adaptive_controller)
                        child._mutation_op_idx = mutated_op_idx
            
            new_population.extend([child1, child2])
        
        # Replace population with new generation, keeping elite
        new_population = new_population[:args.pop_size-1]
        new_population.append(elite_individual)
        population = new_population
        
        # Move to CUDA
        for ind in population:
            ind.mask_normal = ind.mask_normal.cuda()
            ind.mask_reduce = ind.mask_reduce.cuda()
            ind.alphas_normal = ind.alphas_normal.cuda()
            ind.alphas_reduce = ind.alphas_reduce.cuda()

        # Evaluate the fitness of each individual in population Pt

        logging.info(f"Generation {t}: Evaluating fitness for all individuals...")
        for ind in population:
            # Sadece fitness'ı None olanlar (yeni doğanlar) için hesapla
            if ind.fitness is None:
                ind.fitness = evaluate_fitness(valid_queue, model, ind, criterion)

        # Update soft mutation controller with fitness feedback
        for individual in population:
            if hasattr(individual, '_soft_params') and individual.fitness is not None:
                cr, f = individual._soft_params
                parent_fitness = individual._parent_fitness if hasattr(individual, '_parent_fitness') else 0.0
                improved = individual.fitness > parent_fitness
                soft_controller.update_memory(cr, f, improved)
            
            # Update adaptive mutation credit
            if hasattr(individual, '_mutation_op_idx') and individual.fitness is not None:
                pop_fitnesses = [ind.fitness for ind in population if ind.fitness is not None]
                mean_fitness = np.mean(pop_fitnesses) if pop_fitnesses else 0.0
                improved = individual.fitness > mean_fitness
                adaptive_controller.update_credit(individual._mutation_op_idx, improved)
        
        population.sort(key=lambda x: x.fitness if x.fitness is not None else -1, reverse=True)
        best_individual = population[0]
        
        # Calculate and log diversity
        diversity = calculate_population_diversity(population)
        progress = t / args.g_max
        mutation_phase = "Soft" if progress < 0.3 else "Adaptive"
        logging.info(f"Generation {t}: Best Fitness = {best_individual.fitness:.4f}, "
                    f"Diversity = {diversity:.3f}, Mutation = {mutation_phase}")

        logging.info(f"Generation {t} ended.")
        t += 1

    
    # Popülasyonu son bir kez fitness'a göre sırala
    population.sort(key=lambda x: x.fitness if x.fitness is not None else -1, reverse=True)
    best_individual = population[0]

    logging.info("\n" + "="*50)
    logging.info("GENAS SEARCH COMPLETED")
    logging.info(f"Final Best Fitness (Accuracy): {best_individual.fitness:.4f}")
    
    # Genotype'ı bireyin maskesinden türet
    final_genotype = model.parse_from_individual(best_individual)
    logging.info(f'Final Best Genotype: {final_genotype}')


def evaluate_fitness(valid_queue, model, individual, criterion):
    """
    Simpler version: Don't apply additional pruning during evaluation.
    Trust the evolutionary process to select good architectures.
    """
    top1 = utils.AvgrageMeter()
    model.eval()

    with torch.no_grad():
        for step, (input, target) in enumerate(valid_queue):
            input = input.cuda()
            target = target.cuda(non_blocking=True)

            # Use masks directly - they already encode the architecture
            weights_normal = individual.mask_normal.float()
            weights_reduce = individual.mask_reduce.float()
            
            logits = model(input, 
                          weights_normal=weights_normal, 
                          weights_reduce=weights_reduce)
            
            prec1, _ = utils.accuracy(logits, target, topk=(1, 5))
            n = input.size(0)
            top1.update(prec1.item(), n)

    return top1.avg


def train(train_queue, valid_queue, model, architect, criterion, optimizer, lr):
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()

    for step, (input, target) in enumerate(train_queue):
        model.train()
        n = input.size(0)

        input = Variable(input, requires_grad=False).cuda()
        target = Variable(target, requires_grad=False).cuda()

        # get a random minibatch from the search queue with replacement
        input_search, target_search = next(iter(valid_queue))
        input_search = Variable(input_search, requires_grad=False).cuda()
        target_search = Variable(target_search, requires_grad=False).cuda()

        architect.step(input, target, input_search, target_search, lr, optimizer, unrolled=args.unrolled)

        optimizer.zero_grad()
        logits = model(input)
        loss = criterion(logits, target)

        loss.backward()
        nn.utils.clip_grad_norm(model.parameters(), args.grad_clip)
        optimizer.step()

        prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
        objs.update(loss.item(), n)
        top1.update(prec1.item(), n)
        top5.update(prec5.item(), n)

        if step % args.report_freq == 0:
            logging.info('train %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)

    return top1.avg, objs.avg


def infer(valid_queue, model, criterion):
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()
    model.eval()

    for step, (input, target) in enumerate(valid_queue):
        input = Variable(input, volatile=True).cuda()
        target = Variable(target, volatile=True).cuda()

        logits = model(input)
        loss = criterion(logits, target)

        prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
        n = input.size(0)
        objs.update(loss.item(), n)
        top1.update(prec1.item(), n)
        top5.update(prec5.item(), n)

        if step % args.report_freq == 0:
            logging.info('valid %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)

    return top1.avg, objs.avg


if __name__ == '__main__':
    main() 