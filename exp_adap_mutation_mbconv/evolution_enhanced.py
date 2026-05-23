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


# ============================================================================
# ADVANCED MUTATION OPERATORS
# Based on recent NAS literature (2023-2025)
# ============================================================================

class AdaptiveMutationController:
    """
    Adaptive Mutation Operator based on ESE-NAS (2023)
    
    Reference: "Efficient Self-learning Evolutionary Neural Architecture Search"
    Neurocomputing, 2023, DOI: 10.1016/j.neucom.2023.06890
    
    Key Features:
    - Credit Assignment: Learns which operations lead to better performance
    - Model Size Control: Prevents overly complex architectures
    - Adaptive Probabilities: Adjusts operation selection based on success
    """
    
    def __init__(self, num_operations=9, memory_size=10, target_size_ratio=1.2):
        """
        Args:
            num_operations: Number of available operations (9 for GENAS)
            memory_size: Size of credit assignment memory
            target_size_ratio: Target model size relative to baseline
        """
        self.num_operations = num_operations
        self.memory_size = memory_size
        self.target_size_ratio = target_size_ratio
        
        # Credit assignment: [success_count, total_count] for each operation
        self.credit_memory = np.zeros((num_operations, 2))
        
        # Adaptive probability distribution
        self.operation_probs = np.ones(num_operations) / num_operations
        
        # Model size history
        self.size_history = []
    
    def update_credit(self, operation_idx, fitness_improved):
        """
        Update credit for an operation based on mutation success
        
        Args:
            operation_idx: Index of operation that was mutated
            fitness_improved: Whether the mutation improved fitness
        """
        self.credit_memory[operation_idx, 1] += 1  # total count
        if fitness_improved:
            self.credit_memory[operation_idx, 0] += 1  # success count
        
        # Update probabilities based on success rates
        # Use Laplace smoothing to avoid division by zero
        success_rates = (self.credit_memory[:, 0] + 1) / (self.credit_memory[:, 1] + 2)
        
        # Softmax with temperature
        temperature = 0.5
        exp_rates = np.exp(success_rates / temperature)
        self.operation_probs = exp_rates / exp_rates.sum()
    
    def get_operation_probabilities(self, model_size=None):
        """
        Get adjusted probabilities based on model size control
        
        Args:
            model_size: Current model size (optional)
        
        Returns:
            adjusted_probs: Probability distribution over operations
        """
        adjusted_probs = self.operation_probs.copy()
        
        # Model Size Control
        if model_size and self.size_history:
            avg_size = np.mean(self.size_history[-self.memory_size:])
            if model_size > avg_size * self.target_size_ratio:
                # Prefer parameter-free operations (none, skip, pool)
                # Operations 0-3 are typically: none, max_pool, avg_pool, skip
                size_penalty = 0.3
                adjusted_probs[:4] += size_penalty
                adjusted_probs = adjusted_probs / adjusted_probs.sum()
        
        if model_size:
            self.size_history.append(model_size)
            if len(self.size_history) > self.memory_size:
                self.size_history.pop(0)
        
        return adjusted_probs


def mutate_adaptive(individual, adaptive_controller, model_size=None):
    """
    Adaptive Mutation Operator (ESE-NAS 2023)
    
    Uses credit assignment to learn which operations work well.
    Automatically controls model size to prevent over-complexity.
    
    Reference: "Efficient Self-learning Evolutionary Neural Architecture Search"
    Neurocomputing, 2023
    
    Args:
        individual: Individual to mutate
        adaptive_controller: AdaptiveMutationController instance
        model_size: Current model size (optional, for size control)
    
    Returns:
        mutated_op_idx: Index of operation selected (for credit update)
    """
    # Get adaptive probabilities
    op_probs = adaptive_controller.get_operation_probabilities(model_size)
    
    # Select cell type
    target = 'normal' if np.random.rand() < 0.5 else 'reduce'
    mask = individual.mask_normal if target == 'normal' else individual.mask_reduce
    alphas = individual.alphas_normal if target == 'normal' else individual.alphas_reduce
    
    # Select random edge
    row_idx = np.random.randint(0, len(mask))
    
    # Select operation based on adaptive probabilities
    mutated_op_idx = np.random.choice(adaptive_controller.num_operations, p=op_probs)
    
    # Apply mutation
    mask[row_idx].zero_()
    mask[row_idx, mutated_op_idx] = 1.0
    alphas[row_idx] = torch.randn_like(alphas[row_idx]) * 1e-3
    
    individual.fitness = None
    
    return mutated_op_idx


class SoftMutationController:
    """
    Soft Mutation based on SHADE/SADE (2024)
    
    Reference: "Evolutionary Neural Architecture Search for 3D Point Cloud Analysis"
    arXiv 2024, https://arxiv.org/abs/2408.05556
    
    Key Features:
    - Self-adaptive CR (crossover rate) and F (mutation factor)
    - Historical memory of successful parameters
    - Soft perturbation instead of hard replacement
    """
    
    def __init__(self, memory_size=10):
        """
        Args:
            memory_size: Size of historical memory for CR and F
        """
        self.memory_size = memory_size
        
        # Historical memories
        self.MCR = [0.5]  # Mean CR values
        self.MF = [0.5]   # Mean F values
        
        # Success memories
        self.SCR = []
        self.SF = []
    
    def generate_parameters(self):
        """Generate CR and F from historical memories"""
        idx = np.random.randint(0, len(self.MCR))
        
        # CR from normal distribution
        cr = np.clip(np.random.normal(self.MCR[idx], 0.1), 0, 1)
        
        # F from Cauchy distribution
        f = np.clip(np.random.standard_cauchy() * 0.1 + self.MF[idx], 0, 1)
        
        return cr, f
    
    def update_memory(self, cr, f, fitness_improved):
        """Update historical memories based on success"""
        if fitness_improved:
            self.SCR.append(cr)
            self.SF.append(f)
        
        # Update memories periodically
        if len(self.SCR) >= 5:
            if self.SCR:
                mean_cr = np.mean(self.SCR)
                mean_f = np.mean(self.SF)
                
                self.MCR.append(mean_cr)
                self.MF.append(mean_f)
                
                if len(self.MCR) > self.memory_size:
                    self.MCR.pop(0)
                    self.MF.pop(0)
            
            self.SCR = []
            self.SF = []


def mutate_soft(individual, soft_controller):
    """
    Soft Mutation Operator (SHADE 2024)
    
    Instead of hard replacement, uses probabilistic soft mutation with
    self-adaptive parameters CR and F.
    
    Reference: "Evolutionary Neural Architecture Search for 3D Point Cloud Analysis"
    arXiv 2024
    
    Args:
        individual: Individual to mutate
        soft_controller: SoftMutationController instance
    
    Returns:
        cr, f: Parameters used (for success tracking)
    """
    # Generate self-adaptive parameters
    cr, f = soft_controller.generate_parameters()
    
    # Select cell type
    target = 'normal' if np.random.rand() < 0.5 else 'reduce'
    alphas = individual.alphas_normal if target == 'normal' else individual.alphas_reduce
    mask = individual.mask_normal if target == 'normal' else individual.mask_reduce
    
    # Soft mutation on alphas (differential evolution style)
    for i in range(len(alphas)):
        if np.random.rand() < cr:
            # Soft perturbation
            perturbation = torch.randn_like(alphas[i]) * f * 0.01
            alphas[i] = alphas[i] + perturbation
    
    # Probabilistic operation switching
    num_edges_to_mutate = max(1, int(len(mask) * f))
    edges_to_mutate = np.random.choice(len(mask), num_edges_to_mutate, replace=False)
    
    for edge_idx in edges_to_mutate:
        if np.random.rand() < cr:
            # Switch to different operation
            current_op = torch.argmax(mask[edge_idx]).item()
            
            # Prefer operations close to current one (smooth transition)
            op_probs = np.ones(mask.shape[1]) * 0.05
            op_probs[current_op] = 0  # Don't select same operation
            op_probs = op_probs / op_probs.sum()
            
            new_op = np.random.choice(mask.shape[1], p=op_probs)
            mask[edge_idx].zero_()
            mask[edge_idx, new_op] = 1.0
    
    individual.fitness = None
    
    return cr, f


class PeriodicMutationController:
    """
    Periodic Mutation based on MetaNAS (2025)
    
    Reference: "Meta knowledge assisted Evolutionary Neural Architecture Search"
    arXiv 2025, https://arxiv.org/abs/2504.21545
    
    Key Features:
    - Periodic modulation of mutation rate
    - Diversity-based adaptive adjustment
    - Sine wave oscillation for exploration/exploitation balance
    """
    
    def __init__(self, base_rate=0.1, period=5, min_rate=0.05, max_rate=0.3):
        """
        Args:
            base_rate: Base mutation rate
            period: Period for mutation rate oscillation (generations)
            min_rate: Minimum mutation rate
            max_rate: Maximum mutation rate
        """
        self.base_rate = base_rate
        self.period = period
        self.min_rate = min_rate
        self.max_rate = max_rate
        
        self.generation = 0
        self.current_rate = base_rate
        self.diversity_history = []
    
    def get_mutation_rate(self, population_diversity=None):
        """
        Get current mutation rate based on period and diversity
        
        Args:
            population_diversity: Current population diversity measure
        
        Returns:
            current_mutation_rate
        """
        # Periodic modulation
        phase = (self.generation % self.period) / self.period
        periodic_factor = 0.5 * (1 + np.sin(2 * np.pi * phase))
        
        # Base periodic rate
        periodic_rate = self.base_rate + (self.max_rate - self.base_rate) * periodic_factor
        
        # Adaptive adjustment based on diversity
        if population_diversity is not None:
            self.diversity_history.append(population_diversity)
            if len(self.diversity_history) > 10:
                self.diversity_history.pop(0)
            
            # If diversity decreasing, increase mutation
            if len(self.diversity_history) >= 5:
                recent_diversity = np.mean(self.diversity_history[-5:])
                old_diversity = np.mean(self.diversity_history[:5])
                
                if recent_diversity < old_diversity * 0.8:
                    periodic_rate = min(periodic_rate * 1.2, self.max_rate)
        
        self.current_rate = np.clip(periodic_rate, self.min_rate, self.max_rate)
        return self.current_rate
    
    def update_generation(self):
        """Increment generation counter"""
        self.generation += 1


def mutate_periodic(individual, periodic_controller):
    """
    Periodic Mutation Operator (MetaNAS 2025)
    
    Adjusts mutation intensity based on periodic oscillation and population diversity.
    Helps maintain exploration/exploitation balance.
    
    Reference: "Meta knowledge assisted Evolutionary Neural Architecture Search"
    arXiv 2025
    
    Args:
        individual: Individual to mutate
        periodic_controller: PeriodicMutationController instance
    """
    current_rate = periodic_controller.current_rate
    
    # Mutate both cell types with adaptive rate
    for target, mask, alphas in [
        ('normal', individual.mask_normal, individual.alphas_normal),
        ('reduce', individual.mask_reduce, individual.alphas_reduce)
    ]:
        num_edges = len(mask)
        num_to_mutate = max(1, int(num_edges * current_rate))
        
        edges_to_mutate = np.random.choice(num_edges, num_to_mutate, replace=False)
        
        for edge_idx in edges_to_mutate:
            # Resample operation
            mask[edge_idx].zero_()
            op_idx = np.random.randint(0, mask.shape[1])
            mask[edge_idx, op_idx] = 1.0
            alphas[edge_idx] = torch.randn_like(alphas[edge_idx]) * 1e-3
    
    individual.fitness = None


def mutate_guided(individual, population_embeddings=None, num_candidates=3):
    """
    Guided Mutation Operator (PBG 2025)
    
    Generates multiple mutation candidates and selects the most diverse one
    to guide search toward unexplored regions.
    
    Reference: "Population-based guiding for evolutionary neural architecture search"
    Scientific Reports, 2025, DOI: 10.1038/s41598-025-25840-5
    
    Args:
        individual: Individual to mutate
        population_embeddings: List of population embeddings (optional)
        num_candidates: Number of mutation candidates to generate
    
    Note: This is a simplified version. Full implementation would use
          graph neural networks for architecture encoding.
    """
    candidates = []
    diversity_scores = []
    
    for _ in range(num_candidates):
        # Generate mutation candidate
        candidate = copy.deepcopy(individual)
        
        # Standard mutation
        target = 'normal' if np.random.rand() < 0.5 else 'reduce'
        mask = candidate.mask_normal if target == 'normal' else candidate.mask_reduce
        alphas = candidate.alphas_normal if target == 'normal' else candidate.alphas_reduce
        
        row_idx = np.random.randint(0, len(mask))
        mask[row_idx].zero_()
        op_idx = np.random.randint(0, mask.shape[1])
        mask[row_idx, op_idx] = 1.0
        alphas[row_idx] = torch.randn_like(alphas[row_idx]) * 1e-3
        
        candidate.fitness = None
        
        # Compute diversity score (simplified version)
        if population_embeddings:
            # Simple diversity: Hamming distance to population
            candidate_encoding = torch.cat([
                candidate.mask_normal.flatten(),
                candidate.mask_reduce.flatten()
            ])
            
            distances = []
            for pop_encoding in population_embeddings:
                dist = (candidate_encoding != pop_encoding).float().sum().item()
                distances.append(dist)
            
            diversity = np.mean(distances) if distances else 1.0
        else:
            diversity = np.random.rand()  # Random if no embeddings
        
        candidates.append(candidate)
        diversity_scores.append(diversity)
    
    # Select most diverse candidate
    best_idx = np.argmax(diversity_scores)
    best_candidate = candidates[best_idx]
    
    # Copy to original individual
    individual.mask_normal = best_candidate.mask_normal
    individual.mask_reduce = best_candidate.mask_reduce
    individual.alphas_normal = best_candidate.alphas_normal
    individual.alphas_reduce = best_candidate.alphas_reduce
    individual.fitness = None


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def calculate_population_diversity(population):
    """
    Calculate diversity of population based on architecture differences
    
    Args:
        population: List of individuals
    
    Returns:
        diversity: Population diversity score (0-1)
    """
    if len(population) < 2:
        return 0.0
    
    differences = []
    for i in range(len(population)):
        for j in range(i+1, len(population)):
            # Hamming distance
            diff_normal = (population[i].mask_normal != population[j].mask_normal).float().sum().item()
            diff_reduce = (population[i].mask_reduce != population[j].mask_reduce).float().sum().item()
            differences.append(diff_normal + diff_reduce)
    
    # Normalize by maximum possible difference
    if differences:
        max_diff = population[0].mask_normal.numel() + population[0].mask_reduce.numel()
        diversity = np.mean(differences) / max_diff
    else:
        diversity = 0.0
    
    return diversity


def get_population_embeddings(population):
    """
    Get simplified embeddings for population (for guided mutation)
    
    Args:
        population: List of individuals
    
    Returns:
        embeddings: List of flattened mask tensors
    """
    embeddings = []
    for individual in population:
        encoding = torch.cat([
            individual.mask_normal.flatten(),
            individual.mask_reduce.flatten()
        ])
        embeddings.append(encoding)
    
    return embeddings