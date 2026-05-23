import numpy as np
import copy
import torch


def crossover_and_mutation(population, args):
    """
    GENAS Algorithm 1: Crossover and Mutation using Tabular Encoding.
    population: Mevcut SubnetIndividual listesi
    args: Hyperparameters (pop_size, crossover_rate, mutation_rate, tournament_size)
    """
    new_population = []
    pop_size = len(population)
    
    # 1. Elitizm: En iyi bireyi koru [cite: 189]
    # Fitness değerine göre sırala (Accuracy olduğu için büyükten küçüğe)
    population.sort(key=lambda x: x.fitness if x.fitness is not None else -1, reverse=True)
    elite_individual = copy.deepcopy(population[0])
    
    while len(new_population) < pop_size:
        # 2. Tournament Selection [cite: 181]
        parent1 = tournament_selection(population, args.tournament_size)
        parent2 = tournament_selection(population, args.tournament_size)
        
        # 3. Crossover (Offspring generation) 
        if np.random.rand() < args.crossover_rate:
            child1, child2 = row_wise_crossover(parent1, parent2)
        else:
            child1, child2 = copy.deepcopy(parent1), copy.deepcopy(parent2)
            
        # 4. Mutation 
        if np.random.rand() < args.mutation_rate:
            mutate(child1)
        if np.random.rand() < args.mutation_rate:
            mutate(child2)
            
        new_population.extend([child1, child2])

    # Popülasyon boyutunu koru ve eliti ekle
    new_population = new_population[:pop_size-1]
    new_population.append(elite_individual)
    
    return new_population


def tournament_selection(population, t_size):
    # Rastgele t_size kadar birey seç ve en iyisini döndür [cite: 181]
    participants = np.random.choice(population, t_size, replace=False)
    return max(participants, key=lambda x: x.fitness if x.fitness is not None else -1)


def row_wise_crossover(p1, p2):
    """Each row (edge) inherited from one parent."""
    c1 = copy.deepcopy(p1)
    c2 = copy.deepcopy(p2)
    
    for i in range(len(c1.mask_normal)):
        if np.random.rand() < 0.5:
            c1.mask_normal[i] = p2.mask_normal[i].clone()
            c1.alphas_normal[i] = p2.alphas_normal[i].clone()
            c2.mask_normal[i] = p1.mask_normal[i].clone()
            c2.alphas_normal[i] = p1.alphas_normal[i].clone()
    
    # Same for reduce cells
    for i in range(len(c1.mask_reduce)):
        if np.random.rand() < 0.5:
            c1.mask_reduce[i] = p2.mask_reduce[i].clone()
            c1.alphas_reduce[i] = p2.alphas_reduce[i].clone()
            c2.mask_reduce[i] = p1.mask_reduce[i].clone()
            c2.alphas_reduce[i] = p1.alphas_reduce[i].clone()
    
    c1.fitness, c2.fitness = None, None
    return c1, c2


def mutate(individual):
    """Resample a random edge - both mask and alphas"""
    # Select normal or reduce cell
    target = 'normal' if np.random.rand() < 0.5 else 'reduce'
    mask = individual.mask_normal if target == 'normal' else individual.mask_reduce
    alphas = individual.alphas_normal if target == 'normal' else individual.alphas_reduce
    
    # Select random edge
    row_idx = np.random.randint(0, len(mask))
    
    # Resample mask (one-hot encoding)
    mask[row_idx].zero_()
    op_idx = np.random.randint(0, mask.shape[1])
    mask[row_idx, op_idx] = 1.0
    
    # Resample alphas for this edge (small random values like supernet initialization)
    alphas[row_idx] = torch.randn_like(alphas[row_idx]) * 1e-3
    
    individual.fitness = None