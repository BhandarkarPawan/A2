from Population import Population


def genetic_algorithm():
    pop_size = 100
    mutation_rate = 0.01

    target = "To be or not to be."
    pop = Population(target, mutation_rate)
    pop.create_initial_population(pop_size)

    while not pop.finished:
        pop.natural_selection()
        pop.generate_new_population()
        pop.evaluate()
        pop.print_population_status()


if __name__ == "__main__":
    genetic_algorithm()
