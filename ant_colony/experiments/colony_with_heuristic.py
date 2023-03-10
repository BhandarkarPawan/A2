import random
import string
from typing import List

string.printable = "abcdefghijklmnopqrstuvwxyz"


class Ant:
    path: List[int]

    def __init__(self, size: int):
        self.path = self.generate_random_path(size)

    def generate_random_path(self, size: int) -> List[int]:
        path = []

        for _ in range(size):
            path.append(random.randint(0, len(string.printable) - 1))

        return path

    def levenshtein_distance(self, s, t):
        # Create a matrix of zeros with dimensions len(s) + 1 by len(t) + 1
        d = [[0 for j in range(len(t) + 1)] for i in range(len(s) + 1)]

        # Fill in the first row of the matrix
        for i in range(1, len(s) + 1):
            d[i][0] = i

        # Fill in the first column of the matrix
        for j in range(1, len(t) + 1):
            d[0][j] = j

        # Fill in the rest of the matrix
        for j in range(1, len(t) + 1):
            for i in range(1, len(s) + 1):
                if s[i - 1] == t[j - 1]:
                    d[i][j] = d[i - 1][j - 1]
                else:
                    substitution_cost = d[i - 1][j - 1] + 1
                    insertion_cost = d[i][j - 1] + 1
                    deletion_cost = d[i - 1][j] + 1
                    d[i][j] = min(substitution_cost, insertion_cost, deletion_cost)

        return d[len(s)][len(t)]

    def get_fitness(self, target: str) -> int:
        return self.levenshtein_distance(
            "".join(string.printable[node] for node in self.path), target
        )

    def __repr__(self):
        return "".join(string.printable[node] for node in self.path)


class Colony:
    ants: List[Ant]
    evaporation_rate: float
    num_ants: int
    pheromones: List[List[float]]
    pheromone_changes: List[List[float]]
    probabilities: List[List[float]]
    target: str

    def __init__(
        self,
        target,
        num_ants,
        evaporation_rate,
        reward_factor,
        alpha,
        beta,
        max_stagnation,
    ):
        self.target = target
        self.num_ants = num_ants
        self.evaporation_rate = evaporation_rate
        self.reward_factor = reward_factor
        self.size = len(target)
        self.pheromones = self._initialize_matrix(self.size, 1)
        self.pheromone_changes = self._initialize_matrix(self.size, 0)
        self.probabilities = self._initialize_matrix(self.size, 0)
        self.best_ant = None
        self.best_fitness = float("inf")
        self.average_fitness = None
        self.finished = False
        self.perfect_fitness = 0
        self.generations = 0
        self.ants = []
        self.alpha = alpha
        self.beta = beta
        self.max_stagnation = max_stagnation

    def create_initial_colony(self):
        for _ in range(self.num_ants):
            self.ants.append(Ant(self.size))

    def update_pheromones(self):
        self._calculate_pheromone_changes()
        self.generations += 1
        for i in range(self.size):
            for j in range(len(string.printable)):
                self.pheromones[i][j] = (
                    self.evaporation_rate * self.pheromones[i][j]
                    + self.pheromone_changes[i][j]
                )

    def _calculate_pheromone_changes(self):
        self.pheromone_changes = self._initialize_matrix(self.size, 0)

        for k, ant in enumerate(self.ants):
            for i in range(self.size):
                for j in range(len(string.printable)):
                    ant_fitness = 0
                    if ant.path[i] == j:
                        ant_fitness = self.reward_factor / ant.get_fitness(self.target)
                    self.pheromone_changes[i][j] += ant_fitness

    def _initialize_matrix(self, size, value=None):
        matrix = []
        value = random.uniform(0.01, 0.1) if value is None else value
        for _ in range(size):
            matrix.append([value] * len(string.printable))
        return matrix

    def update_paths(self):

        for ant in self.ants:
            self._calculate_probabilities(ant)
            for i in range(self.size):
                ant.path[i] = self._choose_next_node(ant, i)

    def _calculate_probabilities(self, ant):
        for i in range(self.size):
            for j in range(len(string.printable)):
                self.probabilities[i][j] = (
                    self.pheromones[i][j] ** self.alpha
                    * self._heuristic(ant, i, j) ** self.beta
                )

    def _choose_next_node(self, ant, i):
        probabilities = self.probabilities[i]
        total = sum(probabilities)
        probabilities = [p / total for p in probabilities]
        return random.choices(range(len(string.printable)), probabilities)[0]

    def _heuristic(self, ant, i, possible_node):
        current_node = ant.path[i]
        if current_node == possible_node:
            return 0
        return 1 / abs(current_node - possible_node)

    def evaluate(self):
        self.average_fitness = 0
        best_ant_fitness = self.best_fitness
        for ant in self.ants:

            ant_fitness = ant.get_fitness(self.target)
            if ant_fitness < best_ant_fitness:
                self.best_ant = ant
                best_ant_fitness = ant_fitness
            self.average_fitness += ant_fitness
        self.average_fitness /= self.num_ants

        if best_ant_fitness <= self.perfect_fitness:
            self.finished = True

        if best_ant_fitness < self.best_fitness:
            self.best_fitness = best_ant_fitness
            self.stagnation_count = 0
        else:
            self.stagnation_count += 1
            if self.stagnation_count >= self.max_stagnation:
                self.finished = True

    def print_status(self):
        print("\nGeneration: " + str(self.generations))
        print("Average fitness: " + str(self.average_fitness))
        print("Best fitness: " + str(self.best_fitness))
        print("Best Ant: ", self.best_ant)

    def run(self, print_every=100):
        self.create_initial_colony()
        while not self.finished:
            self.update_pheromones()
            self.update_paths()
            self.evaluate()

            if self.generations % print_every == 0:
                self.print_status()

        if self.stagnation_count >= self.max_stagnation:
            print("\nMax stagnation reached. Stopping...")


def ant_colony_algorithm():
    num_ants = 100
    evaporation_rate = 0.7
    reward_factor = 1
    alpha = 1
    beta = 1
    max_stagnation = 1000

    target = "helloworld"
    colony = Colony(
        target=target,
        num_ants=num_ants,
        evaporation_rate=evaporation_rate,
        reward_factor=reward_factor,
        alpha=alpha,
        beta=beta,
        max_stagnation=max_stagnation,
    )

    colony.run(1)


if __name__ == "__main__":
    ant_colony_algorithm()
