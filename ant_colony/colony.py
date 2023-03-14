import random
import string
from typing import List

from ant import Ant

# Limit the printable characters to the lowercase alphabet
# string.printable = "abcdefghijklmnopqrstuvwxyz"
char_to_int = {char: i for i, char in enumerate(string.printable)}


class Colony:
    ants: List[Ant]
    evaporation_rate: float
    num_ants: int
    pheromones: List[List[float]]
    pheromone_changes: List[List[float]]
    probabilities: List[List[float]]
    target: str
    size: int
    best_ant: Ant
    best_fitness: int
    average_fitness: float
    finished: bool
    perfect_fitness: int
    generations: int
    alpha: float
    beta: float
    max_stagnation: int

    def __init__(
        self,
        target: str,
        num_ants: int,
        evaporation_rate: float,
        reward_factor: float,
        alpha: float,
        beta: float,
        max_stagnation: int,
    ) -> None:
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

    def create_initial_colony(self) -> None:
        """
        Create the initial colony of ants.
        Each ant is initialized with a random path.
        """
        for _ in range(self.num_ants):
            self.ants.append(Ant(self.size))

    def update_pheromones(self) -> None:
        """
        Update the pheromone matrix based on the ants' paths.

        The pheromone matrix is updated by first calculating the
        pheromone changes based on the ants' paths, then updating
        the pheromone matrix by adding the pheromone changes.
        """
        self._calculate_pheromone_changes()
        self.generations += 1
        for i in range(self.size):
            for j in range(len(string.printable)):
                self.pheromones[i][j] = (
                    self.evaporation_rate * self.pheromones[i][j]
                    + self.pheromone_changes[i][j]
                )

    def _calculate_pheromone_changes(self):
        """
        Calculate the pheromone changes based on the ants' paths.

        The pheromone changes are calculated by iterating through
        each ant and each character in the target string. If the
        ant's path matches the character in the target string, then
        the pheromone change is calculated by dividing the reward
        factor by the ant's fitness.
        """
        self.pheromone_changes = self._initialize_matrix(self.size, 0)

        for k, ant in enumerate(self.ants):
            for i in range(self.size):
                for j in range(len(string.printable)):
                    ant_fitness = 0
                    if ant.path[i] == j:
                        ant_fitness = self.reward_factor / ant.get_fitness(self.target)
                    self.pheromone_changes[i][j] += ant_fitness

    def _initialize_matrix(self, size, value=None):
        """
        Initialize a matrix of the given size with the given value.

        If no value is given, then the matrix is initialized with
        random values between 0.01 and 0.1.
        """
        matrix = []
        value = random.uniform(0.01, 0.1) if value is None else value
        for _ in range(size):
            matrix.append([value] * len(string.printable))
        return matrix

    def update_paths(self):
        """
        Update the ants' paths based on the pheromone matrix.

        The ants' paths are updated by first calculating the
        probabilities of each character in the target string, then
        choosing the next character in the path based on the
        probabilities.
        """
        self._calculate_probabilities()
        for ant in self.ants:
            for i in range(self.size):
                ant.path[i] = self._choose_next_node(i)

    def _calculate_probabilities(self):
        """
        Calculate the probabilities of each character in the target string.

        The probabilities are calculated by iterating through each
        character in the target string and calculating the probability
        of each character in the printable string. The probability of
        each character is calculated by dividing the pheromone value
        of the character by the total pheromone value of all the options
        for the current character.
        """
        self.probabilities = self._initialize_matrix(self.size, 0)
        for i in range(self.size):
            total = sum(self.pheromones[i])
            for j in range(len(string.printable)):
                self.probabilities[i][j] = self.pheromones[i][j] ** self.alpha / total

        self.probabilities = self._initialize_matrix(self.size, 0)
        for i in range(self.size):
            for j in range(len(string.printable)):
                self.probabilities[i][j] = (
                    self.pheromones[i][j] ** self.alpha
                    * self._heuristic(i, j) ** self.beta
                )

        for i in range(self.size):
            total = sum(self.probabilities[i])
            for j in range(len(string.printable)):
                self.probabilities[i][j] /= total

    def _choose_next_node(self, i):
        """
        Choose the next character in the ant's path based on the probabilities.

        The next character is chosen by generating a random number
        between 0 and 1 and iterating through the probabilities until
        the cumulative probability is greater than the random number.
        This is known as the roulette wheel selection method.
        """

        cumulative_probability = 0

        r = random.random()
        for j in range(len(string.printable)):
            cumulative_probability += self.probabilities[i][j]
            if r <= cumulative_probability:
                return j

    def evaluate(self):
        """
        Evaluate the ants' paths and update the best ant and average fitness.

        The ants' paths are evaluated by iterating through each ant
        and calculating the ant's fitness. If the ant's fitness is
        less than the best fitness, then the best ant is updated.
        """
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

    def _heuristic(self, current_node: int, next_node: int) -> int:
        """
        Calculate the heuristic value for the given character.

        The heuristic value is calculated by taking the absolute
        value of the difference between the current character and
        the target character.
        """
        if self.target[current_node] == string.printable[next_node]:
            return 1
        return 1 / abs(next_node - char_to_int[self.target[current_node]])

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
