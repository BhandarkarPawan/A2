import random
import string
from typing import List

# Limit the printable characters to the lowercase alphabet
# string.printable = "abcdefghijklmnopqrstuvwxyz"


class Ant:
    path: List[int]

    def __init__(self, size: int):
        self.path = self.generate_random_path(size)

    def generate_random_path(self, size: int) -> List[int]:
        path = []
        for _ in range(size):
            path.append(random.randint(0, len(string.printable) - 1))
        return path

    def _levenshtein_distance(self, s: str, t: str) -> int:
        """
        Calculate the Levenshtein distance between two strings.

        The Levenshtein distance is the number of characters that need to be
        substituted, inserted, or deleted, to transform s into t.
        """
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
        path_string = "".join(string.printable[node] for node in self.path)
        return self._levenshtein_distance(path_string, target)

    def __repr__(self):
        return "".join(string.printable[node] for node in self.path)
