
import numpy as np
from random import random
from functorch.experimental.control_flow import map

import torch
from torch import nn


class ScoresNotComputedException(Exception):
    def __init__(self, *args: object) -> None:
        super().__init__(*args)

class GeneticAlgorithm(nn.Module):
    def __init__(self, pop_len: int, n_genes: int, device=None, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        if "inicialization" not in kwargs.keys():
            self.population = torch.rand((pop_len, n_genes))
        elif kwargs["inicialization"].upper() == "ONES":
            self.population = torch.ones((pop_len, n_genes))
        elif kwargs["inicialization"].upper() == "ZERO":
            self.population = torch.zeros((pop_len, n_genes))
        elif kwargs["inicialization"].upper() == "RANDOM":
            self.population = torch.rand((pop_len, n_genes))
        
        self.device = device if device else torch.device("cpu")
        self.population = self.population.to(self.device)

        self.softmax = nn.Softmax(dim=-1)

        self.__cross_fn = torch.func.vmap(self.__cross, randomness="different")
        
    def __choice(self):
        loc = torch.rand((self.population.shape[0], 1), device=self.device)
        index = torch.sum(self.scores < loc, dim=-1)
        
        return self.population[index]
    
    def __cross(self, ind_1, ind_2):
        spread = np.exp(-1 * self.gen / 10)
        mut_ch = 0.1 * np.exp(-1 * self.gen / 10)

        crossing_factor = (random() - 0.5) * spread
        delta = ind_2 - ind_1

        son = ind_1 + delta * crossing_factor

        if random() <= mut_ch:
            mutation = torch.rand((ind_1.shape[0]), device=self.device)
            mutation = mutation / torch.sqrt(torch.sum(torch.pow(mutation, 2))) * spread

            son = son + mutation
        
        return son

    def new_population(self, func, generation = 0):
        self.__update_scores(func(self.population))
        self.gen = generation
        
        new_population = torch.zeros(self.population.shape)
        new_population = self.__cross_fn(self.__choice(), self.__choice())
        self.population = new_population

    def __update_scores(self, scores: torch.Tensor):
        if scores.dim() == 2 and scores.shape[0] == 1:
            scores = scores.squeeze()

        if torch.sum(scores) != 1:
            scores = scores / torch.sum(scores)

        acc_scores = torch.zeros(scores.shape, device=self.device)
        acc = 0
        for i, score in enumerate(scores):
            acc += score
            acc_scores[i] = acc
        
        acc_scores[-1] = 1
        
        self.scores = acc_scores
    
    def best_individual(self):
        if hasattr(self, "scores"):
            index = torch.max(ga.scores, dim=0).indices
            return self.population[index]

        else:
            raise ScoresNotComputedException(
                "This population does not have scores computed. Please execute train to compute it."
            )


if __name__ == "__main__":
    from time import time
    from tqdm import tqdm
    import matplotlib.pyplot as plt


    device = torch.device("cpu")
    ga = GeneticAlgorithm(100, 5, device=device)
    coefs = torch.rand((5), device=device)

    best_scores = list()
    def func(pop):
        x = torch.matmul(pop, coefs)
        best_scores.append(torch.max(x))

        return x

    start = time()
    for i in tqdm(range(100)):
        ga.new_population(func, i)
    
    print(ga.best_individual())
    print(coefs)

    print("\n\nTempo de execução: {} segundos".format(time() - start))
    print(f"Device: {device}")

    plt.plot(best_scores)
    plt.title("Aptidão do melhor indivíduo por geração")
    plt.show()