import random
from typing import List, Tuple

from dataset import Dataset
from individual import Individual


class Population:
    def __init__(self, dataset: Dataset, mutation_rate, stagnation_limit=1000, cv=10):
        self.dataset = dataset
        self.population: List[Individual] = []
        self.generations = 0
        self.mutation_rate = mutation_rate
        self.best_individual = None
        self.finished = False
        self.perfect_score = 1
        self.max_fitness = 0
        self.average_fitness = 0
        self.mating_pool = []
        self.stagnation_limit = stagnation_limit
        self.generations_without_improvement = 0
        self.cv = cv

    def create_initial_population(self, pop_size):
        for _ in range(pop_size):
            ind = Individual(self.dataset, self.cv)
            ind.calculate_fitness()

            if ind.fitness > self.max_fitness:
                self.max_fitness = ind.fitness
                self.best_individual = ind

            self.average_fitness += ind.fitness
            self.population.append(ind)

        self.average_fitness /= len(self.population)

    def natural_selection(self):
        self.mating_pool = []

        for index, ind in enumerate(self.population):
            prob = int(round(ind.fitness * 100))
            self.mating_pool.extend([index for i in range(prob)])

    def generate_new_population(self):
        new_population = []
        pop_size = len(self.population)
        self.average_fitness = 0

        for i in range(pop_size):
            partner_a, partner_b = self.selection()

            offspring = partner_a.crossover(partner_b)
            offspring.mutate(self.mutation_rate)
            offspring.calculate_fitness()

            self.average_fitness += offspring.fitness
            new_population.append(offspring)

        self.population = new_population
        self.generations += 1
        self.average_fitness /= pop_size

    def selection(self) -> Tuple[Individual, Individual]:
        pool_size = len(self.mating_pool)

        index_a = random.randint(0, pool_size - 1)
        index_b = random.randint(0, pool_size - 1)

        return (
            self.population[self.mating_pool[index_a]],
            self.population[self.mating_pool[index_b]],
        )

    def evaluate(self):
        best_fitness = 0
        best_individual = None

        for ind in self.population:
            if ind.fitness > best_fitness:
                best_fitness = ind.fitness
                best_individual = ind

        if best_fitness > self.max_fitness:
            self.max_fitness = best_fitness
            self.best_individual = best_individual
            self.generations_without_improvement = 0
        else:
            self.generations_without_improvement += 1

        if self.max_fitness >= self.perfect_score:
            self.finished = True

    def print_population_status(self):
        print("\nGeneration: " + str(self.generations))
        print("Average fitness: " + str(self.average_fitness))
        print("Max fitness: " + str(self.max_fitness))

        if self.generations_without_improvement >= self.stagnation_limit:
            print("Stagnation limit reached")
            self.finished = True
        elif self.max_fitness >= self.perfect_score:
            print("Perfect score reached")
            self.finished = True
