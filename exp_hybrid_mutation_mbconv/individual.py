import torch
import numpy as np
import copy

class SubnetIndividual:


    def __init__(self, alphas_normal, alphas_reduce):
        self.alphas_normal = alphas_normal.clone().detach()
        self.alphas_reduce = alphas_reduce.clone().detach()
        self.mask_normal = self._create_one_hot_mask(self.alphas_normal)
        self.mask_reduce = self._create_one_hot_mask(self.alphas_reduce)
        self.fitness = None


    def _create_one_hot_mask(self, alphas):
        n_edges, n_ops = alphas.shape
        mask = torch.zeros_like(alphas)
        
        for i in range(n_edges):
            # Select operation with highest alpha value initially
            selected_op = torch.argmax(alphas[i]).item()
            mask[i, selected_op] = 1.0
            
        return mask


    def update_mask_from_alphas(self):
        # Yerel optimizasyon sonrası alpha vektöründeki en büyük değeri 1, diğerlerini 0 yap
        self.mask_normal.data.fill_(0)
        indices_n = torch.argmax(self.alphas_normal, dim=-1)
        for i, idx in enumerate(indices_n):
            self.mask_normal[i, idx] = 1.0

        self.mask_reduce.data.fill_(0)
        indices_r = torch.argmax(self.alphas_reduce, dim=-1)
        for i, idx in enumerate(indices_r):
            self.mask_reduce[i, idx] = 1.0