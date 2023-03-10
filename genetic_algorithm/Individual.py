import random
import string


class Individual:
    def __init__(self, size):
        self.genes = self.generate_random_genes(size)

    def generate_random_genes(self, size):
        genes = []

        for i in range(size):
            genes.append(random.choice(string.printable))

        return genes

    # Fitness function 1: it returns a floating point of "correct" characters
    def calculate_fitness(self, target):
        index = 0
        score = 0

        for gene in self.genes:
            if gene == target[index]:
                score += 1
            index += 1

        self.fitness = score / len(target)

    def crossover(self, partner):
        ind_length = len(self.genes)
        child = Individual(ind_length)

        midpoint = random.randint(0, ind_length)

        child.genes = self.genes[:midpoint] + partner.genes[midpoint:]

        return child

    def mutate(self, mutation_rate):
        for i in range(len(self.genes)):
            if random.uniform(0, 1) < mutation_rate:
                self.genes[i] = random.choice(string.printable)

    def __str__self(self):
        return "".join(self.genes) + "-> fitness: " + str(self.fitness)
