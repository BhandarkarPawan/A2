import random
from typing import List, Tuple

from Individual import Individual


class Population:
    def __init__(self, target, mutation_rate):
        self.population: List[Individual] = []
        self.generations = 0
        self.target = target
        self.mutation_rate = mutation_rate
        self.best_individual = None
        self.finished = False
        self.perfect_score = 1
        self.max_fitness = 0
        self.average_fitness = 0
        self.mating_pool = []

    def create_initial_population(self, size):
        for i in range(size):
            ind = Individual(len(self.target))
            ind.calculate_fitness(self.target)

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
            offspring.calculate_fitness(self.target)

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

        for ind in self.population:
            if ind.fitness > best_fitness:
                best_fitness = ind.fitness
                self.best_individual = ind
                self.max_fitness = best_fitness

        if best_fitness >= self.perfect_score:
            self.finished = True

    def print_population_status(self):
        print("\nGeneration: " + str(self.generations))
        print("Average fitness: " + str(self.average_fitness))
        print("Max fitness: " + str(self.max_fitness))
