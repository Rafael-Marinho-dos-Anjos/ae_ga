
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
            self.population = torch.rand((pop_len, n_genes)) * 2 - 1
        elif kwargs["inicialization"].upper() == "ONES":
            self.population = torch.ones((pop_len, n_genes))
        elif kwargs["inicialization"].upper() == "ZERO":
            self.population = torch.zeros((pop_len, n_genes))
        elif kwargs["inicialization"].upper() == "RANDOM":
            self.population = torch.rand((pop_len, n_genes))
        elif kwargs["inicialization"].upper() == "RANDOM_W_NEGATIVES":
            self.population = torch.rand((pop_len, n_genes)) * 2 - 1
        
        self.device = device if device else torch.device("cpu")
        self.population = self.population.to(self.device)

        self.__cross_fn = torch.func.vmap(self.__cross, randomness="different")
        
    def __choice(self):
        loc = torch.rand((self.population.shape[0], 1), device=self.device)
        index = torch.sum(self.acc_scores < loc, dim=-1)
        
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

    def new_population(self, func, generation=0, n_survivors=0):
        self.__update_scores(func(self.population))
        self.gen = generation
        
        new_population = torch.zeros(self.population.shape)
        new_population = self.__cross_fn(self.__choice(), self.__choice())

        if n_survivors > 0:
            survivors = torch.topk(self.scores, k=n_survivors).indices
            new_population[0: n_survivors] = self.population[survivors]

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
        
        self.scores = scores
        self.acc_scores = acc_scores
    
    def best_individual(self):
        if hasattr(self, "scores"):
            index = torch.argmax(self.scores, dim=0)
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
    coefs = torch.rand((6), device=device) * 2 - 1

    best_scores = list()

    softmin = torch.nn.Softmin()
    operator = torch.vmap(
        lambda x: x[0]*coefs[0] + (x[1]**2)*coefs[1] + (x[2]**3)*coefs[2] + (x[3]**4)*coefs[3] + (x[4]**5)*coefs[4] + coefs[5]
    )
    def func(pop):
        x = operator(pop)
        best_scores.append(torch.max(x).item())
        x = softmin(x)

        return x

    start = time()
    for i in tqdm(range(1000)):
        ga.new_population(func, i, 3)
    
    print(ga.best_individual())
    print(coefs)

    print("\n\nTempo de execução: {} segundos".format(time() - start))
    print(f"Device: {device}")

    plt.plot(best_scores)
    plt.title("Aptidão do melhor indivíduo por geração")
    plt.show()
