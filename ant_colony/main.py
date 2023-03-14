from colony import Colony


def ant_colony_algorithm():
    num_ants = 100
    evaporation_rate = 0.7
    reward_factor = 1
    alpha = 1
    beta = 1
    max_stagnation = 1000

    target = "To be or not to be"
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
