import random
import numpy as np
import matplotlib.pyplot as plt


NUM_DIRECTIONS = 4
CYCLE_TIME = 120
POPULATION_SIZE = 30
GENERATIONS = 100
MUTATION_RATE = 0.5
NUM_RANDOM_INDIVIDUALS = 5

arrival_rates = [0.5, 0.3, 0.4, 0.6]
departure_rate = 1.0

def generate_chromosome():
    splits = sorted(random.sample(range(1, CYCLE_TIME), NUM_DIRECTIONS - 1))
    green_times = [splits[0]] + [splits[i] - splits[i - 1] for i in range(1, NUM_DIRECTIONS - 1)]
    green_times.append(CYCLE_TIME - splits[-1])
    return green_times

def fitness(chromosome):
    total_waiting = 0
    for i in range(NUM_DIRECTIONS):
        arriving = arrival_rates[i] * CYCLE_TIME
        passing = departure_rate * chromosome[i]
        waiting = max(0, arriving - passing)
        total_waiting += waiting
    return -total_waiting

def selection(population, fitnesses):
    candidates = random.sample(range(len(population)), 3)
    best = max(candidates, key=lambda idx: fitnesses[idx])
    return population[best]

def crossover(p1, p2):
    child = [random.choice([p1[i], p2[i]]) for i in range(NUM_DIRECTIONS)]
    total = sum(child)
    scale = CYCLE_TIME / total if total > 0 else 1
    child = [max(1, int(round(g * scale))) for g in child]
    diff = CYCLE_TIME - sum(child)
    child[random.randint(0, NUM_DIRECTIONS - 1)] += diff
    return child

def mutate(chromosome):
    new_chromo = chromosome[:]
    for _ in range(random.randint(1, 2)):
        i, j = random.sample(range(NUM_DIRECTIONS), 2)
        max_delta = min(10, new_chromo[i] - 1)
        if max_delta <= 0:
            continue
        delta = random.randint(1, max_delta)
        new_chromo[i] -= delta
        new_chromo[j] += delta
    return new_chromo

def genetic_algorithm():
    population = [generate_chromosome() for _ in range(POPULATION_SIZE)]
    best_solution = None
    best_fitness = float('-inf')

    bests = []
    avgs = []

    for gen in range(GENERATIONS):
        fitnesses = [fitness(chromo) for chromo in population]
        avg_fitness = sum(fitnesses) / len(fitnesses)

        elite_idx = np.argmax(fitnesses)
        elite = population[elite_idx]
        elite_fit = fitnesses[elite_idx]

        new_population = [elite]

        while len(new_population) < POPULATION_SIZE - NUM_RANDOM_INDIVIDUALS:
            parent1 = selection(population, fitnesses)
            parent2 = selection(population, fitnesses)
            child = crossover(parent1, parent2)
            if random.random() < MUTATION_RATE:
                child = mutate(child)
            new_population.append(child)

        for _ in range(NUM_RANDOM_INDIVIDUALS):
            new_population.append(generate_chromosome())

        population = new_population
        fitnesses = [fitness(chromo) for chromo in population]
        gen_best_idx = np.argmax(fitnesses)
        gen_best_fit = fitnesses[gen_best_idx]

        if gen_best_fit > best_fitness:
            best_fitness = gen_best_fit
            best_solution = population[gen_best_idx]

        bests.append(-gen_best_fit)
        avgs.append(-avg_fitness)

        if (gen + 1) % 10 == 0 or gen == 0:
            print(f"Generation {gen + 1:3} | Best Green Times: {best_solution} | Waiting Cars: {round(-best_fitness, 2)}")

    return best_solution, -best_fitness, bests, avgs

# Run the GA
best_solution, best_waiting_time, bests, avgs = genetic_algorithm()

# Plot convergence
plt.figure(figsize=(10, 6))
plt.plot(bests, label='Best Fitness')
plt.plot(avgs, label='Average Fitness', linestyle='--')
plt.xlabel('Generation')
plt.ylabel('Total Waiting Cars')
plt.title('GA Convergence - Traffic Signal Optimization')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Final result
print(" Best Green Light Timings [N, E, S, W]:", best_solution)
print(" Minimum Total Waiting Cars:", round(best_waiting_time, 2))