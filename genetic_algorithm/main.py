from dataset import Dataset
from individual import evaluate_selection
from population import Population


def genetic_algorithm():
    pop_size = 10
    mutation_rate = 0.01
    cv = 10

    dataset = Dataset()

    acc, f1 = evaluate_selection(*dataset.get_data(), cv=cv)
    print("Initial Model Performance")
    print(f"Accuracy: {acc}")
    print(f"F1 Score: {f1}")

    pop = Population(dataset, mutation_rate, stagnation_limit=10, cv=cv)
    pop.create_initial_population(pop_size)

    while not pop.finished:
        pop.natural_selection()
        pop.generate_new_population()
        pop.evaluate()
        pop.print_population_status()

    best_genes = pop.best_individual.genes
    dataset.select_features(best_genes)
    selected_features = dataset.get_selected_feature_names()

    print(f"\nSelected Feature Count: {sum(best_genes)}")
    print(f"Selected Features: {selected_features}")

    acc, f1 = evaluate_selection(*dataset.get_data(), cv=cv)
    print(f"\nFinal Model Performance")
    print(f"Accuracy: {acc}")
    print(f"F1 Score: {f1}")


if __name__ == "__main__":
    genetic_algorithm()
