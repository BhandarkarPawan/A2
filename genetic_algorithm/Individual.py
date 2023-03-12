import random

import pandas as pd
from dataset import Dataset
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.tree import DecisionTreeClassifier


def evaluate_selection(X: pd.DataFrame, y: pd.Series, cv: int = 10):
    average_accuracy = 0
    average_f1 = 0

    for i, (train_index, test_index) in enumerate(
        StratifiedKFold(n_splits=cv, shuffle=True).split(X, y)
    ):

        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        model = DecisionTreeClassifier()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        average_accuracy += accuracy
        average_f1 += f1

    average_accuracy /= cv
    average_f1 /= cv

    return average_accuracy, average_f1


class Individual:
    genes: list
    fitness: float
    dataset: Dataset
    cv: int

    def __init__(self, dataset: Dataset, cv: int = 10):
        size = dataset.num_features
        self.genes = self.generate_random_genes(size)
        self.fitness = 0
        self.dataset = dataset
        self.cv = cv

    def generate_random_genes(self, size: int) -> list:
        genes = []

        for _ in range(size):
            genes.append(random.randint(0, 1))

        return genes

    def calculate_fitness(self):
        self.dataset.select_features(self.genes)
        X, y = self.dataset.get_data()

        if X.empty:
            self.fitness = 0
            return

        _, f1 = evaluate_selection(X, y)
        self.fitness = f1

    def crossover(self, partner: "Individual") -> "Individual":
        ind_length = len(self.genes)
        child = Individual(self.dataset)
        midpoint = random.randint(0, ind_length - 1)
        child.genes = self.genes[:midpoint] + partner.genes[midpoint:]

        return child

    def mutate(self, mutation_rate):
        for i in range(len(self.genes)):
            if random.random() < mutation_rate:
                self.genes[i] = 1 if self.genes[i] == 0 else 0

    def __str__(self):
        return "".join(self.genes) + "-> fitness: " + str(self.fitness)
