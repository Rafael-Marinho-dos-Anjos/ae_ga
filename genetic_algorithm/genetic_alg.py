
import numpy as np
from random import random

import torch
from torch import nn


class GeneticAlgorithm(nn.Module):
    def __init__(self, pop_len: int, n_genes: int, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        if "inicialization" not in kwargs.keys():
            self.population = torch.rand((pop_len, n_genes))
        elif kwargs["inicialization"].upper() == "ONES":
            self.population = torch.ones((pop_len, n_genes))
        elif kwargs["inicialization"].upper() == "ZERO":
            self.population = torch.zeros((pop_len, n_genes))
        elif kwargs["inicialization"].upper() == "RANDOM":
            self.population = torch.rand((pop_len, n_genes))

        self.softmax = nn.Softmax(dim=-1)
        
    def objective_func(self, coefs):
        x = torch.matmul(self.population, coefs)
        self.scores = x

        x = self.softmax(x)

        return x

    def new_population(self, coefs, generation = 0):
        scores = self.objective_func(coefs)
        spread = np.exp(-1 * generation / 10)
        mut_ch = 0.1 * np.exp(-1 * generation / 10)

        def __choice():
            loc = random()
            acc = 0
            index = -1
            
            while acc < loc:
                index += 1
                acc += scores[index]
            
            return self.population[index]
        
        def __cross(ind_1, ind_2):
            crossing_factor = (random() - 0.5) * spread
            delta = ind_2 - ind_1

            son = ind_1 + delta * crossing_factor

            if random() <= mut_ch:
                mutation = torch.rand((ind_1.shape[0]))
                mutation = mutation / torch.sqrt(torch.sum(torch.pow(mutation, 2))) * spread

                son = son + mutation
            
            return son
        
        self.population = torch.stack([__cross(__choice(), __choice()) for i in range(self.population.shape[0])])

if __name__ == "__main__":
    ga = GeneticAlgorithm(100, 5)
    coefs = torch.rand((5))

    for i in range(100):
        ga.new_population(coefs, i)
        print("", i, torch.max(ga.scores))
    
    index = torch.max(ga.scores, dim=0).indices
    print(ga.population[index])
    print(coefs)