# DynoNAS: Dynamic Mutations for Hybrid Neural Architecture Search

Implemented on the DARTS codebase.

---

## Quick start

### Dependencies

```bash
pip install torch torchvision numpy
```

### Test With Trained Model

```bash
python dynonas/test_final.py \
    --model_path model/weights.pt \
    --genotype_file model/best_genotype.txt
```

---


## Detailed Implementation

First, start with architecture search. Different evolutionary operators can be selected with mutation_type parameter. Sample commands are shown in step 1. After finding best genotype, train the full network as shown in step 2. Finally you can test the model by changing the files under model folder.

### 1. Full CIFAR-10 genotype search

```bash
# Standard mutation
python dynonas/dynonas_search.py --data ./data --dataset cifar10

# Soft mutation
python dynonas/dynonas_search.py --data ./data --mutation_type soft

# Adaptive mutation
python dynonas/dynonas_search.py --data ./data --mutation_type adaptive

# Zero mutation
python dynonas/dynonas_search.py --data ./data --mutation_type zero

# Dynamic mutation
python dynonas/dynonas_search.py --data ./data --mutation_type hybrid
```

Output: `search-dynonas-<timestamp>/best_genotype.txt`

### 2. Train full architecture (600 epochs)

```bash
python train_final.py \
    --data ./data \
    --genotype_file search-dynonas-<timestamp>/best_genotype.txt \
    --auxiliary --cutout
```

### 3. Test With Trained Model

```bash
python dynonas/test_final.py \
    --model_path model/best_weights.pt \
    --genotype_file model/best_genotype.txt
```

### CIFAR-100 search + retraining

```bash
python3 dynonas/dynonas_search.py --data ./data --dataset cifar100
python3 train_final.py --data ./data --dataset cifar100 \
    --genotype_file search-dynonas-<timestamp>/best_genotype.txt \
    --auxiliary --cutout
```

---

