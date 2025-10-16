import numpy as np
import matplotlib.pyplot as plt


def power_consumption(params):
    # params is a numpy array [voltage, current, frequency]
    voltage, current, frequency = params
    
    noise = np.random.normal(0, 0.1)
    
    target_voltage = 5.0
    target_current = 3.0
    target_frequency = 2.0
    return (voltage - target_voltage)**2 + (current - target_current)**2 + (frequency - target_frequency)**2 + voltage * current + frequency * 0.5 + 10 + noise

# Cuckoo Search Optimization Algorithm
def cuckoo_search(objective_func, search_space, num_nests=25, num_iterations=100, pa=0.25, alpha=0.5):
    """
    Cuckoo Search Optimization
    objective_func: The function to minimize
    search_space: A tuple of (min_bound, max_bound) for each parameter
    num_nests: Number of nests (solutions)
    num_iterations: Number of iterations
    pa: Probability of alien egg discovery
    alpha: Step size scaling factor
    """
    num_params = len(search_space)
    min_bounds = np.array([s[0] for s in search_space])
    max_bounds = np.array([s[1] for s in search_space])

    # Initialize nests randomly within the search space
    nests = min_bounds + (max_bounds - min_bounds) * np.random.rand(num_nests, num_params)
    best_nest = nests[0].copy()
    best_fitness = objective_func(best_nest)

    fitness_history = []

    for iter in range(num_iterations):
        
        step_size = alpha * (nests[np.random.randint(num_nests)] - nests[np.random.randint(num_nests)])
        new_nest = nests[np.random.randint(num_nests)] + step_size

        # Apply bounds
        new_nest = np.clip(new_nest, min_bounds, max_bounds)

        # Evaluate the new nest's fitness
        new_fitness = objective_func(new_nest)

        # Choose a random nest to compare with
        j = np.random.randint(num_nests)

        # Replace the random nest if the new nest is better
        if new_fitness < objective_func(nests[j]):
            nests[j] = new_nest
            # Update best nest if necessary
            if new_fitness < best_fitness:
                best_fitness = new_fitness
                best_nest = new_nest.copy()

        # Discover and abandon poor nests
        # Randomly select nests to be abandoned
        abandoned_nests_indices = np.random.rand(num_nests) < pa
        num_abandoned = np.sum(abandoned_nests_indices)

        # Generate new nests for the abandoned ones
        if num_abandoned > 0:
            nests[abandoned_nests_indices] = min_bounds + (max_bounds - min_bounds) * np.random.rand(num_abandoned, num_params)

            # Re-evaluate fitness for potentially new best nest after abandonment
            current_best_fitness = objective_func(best_nest)
            for i in range(num_nests):
                fit = objective_func(nests[i])
                if fit < current_best_fitness:
                    current_best_fitness = fit
                    best_nest = nests[i].copy()
                    best_fitness = current_best_fitness


        fitness_history.append(best_fitness)

        if (iter + 1) % 10 == 0 or iter == 0:
             print(f"Iteration {iter + 1}/{num_iterations} - Best Power Consumption: {best_fitness:.4f}")


    return best_nest, best_fitness, fitness_history

# --- Main Execution ---
if __name__ == "__main__":
    # Define the search space for the parameters (e.g., [min_voltage, max_voltage], [min_current, max_current], [min_frequency, max_frequency])
    search_space = [(1, 10), (0.5, 5), (1, 4)] # Added frequency search space

    # Run the Cuckoo Search Optimization
    optimal_params, min_power, history = cuckoo_search(power_consumption, search_space)

    print("\n--- Optimization Results ---")
    print("Optimal Parameters:", optimal_params)
    print("Minimum Power Consumption:", min_power)

    # --- Visualization ---
    plt.figure(figsize=(10, 6))
    plt.plot(history)
    plt.xlabel("Iteration")
    plt.ylabel("Best Power Consumption")
    plt.title("Cuckoo Search Optimization Convergence")
    plt.grid(True)
    plt.show()

    # Visualize a slice of the objective function landscape (fixing frequency)
    if len(search_space) >= 2:
       
        fixed_frequency = optimal_params[2] 

        x = np.linspace(search_space[0][0], search_space[0][1], 100) # Voltage
        y = np.linspace(search_space[1][0], search_space[1][1], 100) # Current
        X, Y = np.meshgrid(x, y)

        # Calculate Z (power consumption) for the fixed frequency
        Z = np.array([[power_consumption([xi, yi, fixed_frequency]) for yi in y] for xi in x])

        plt.figure(figsize=(10, 8))
        contour = plt.contourf(X, Y, Z, levels=50, cmap='viridis')
        plt.colorbar(contour, label='Power Consumption')
        plt.scatter(optimal_params[0], optimal_params[1], color='red', marker='*', s=200, label='Optimal Solution (at fixed frequency)')
        plt.xlabel("Voltage")
        plt.ylabel("Current")
        plt.title(f"Power Consumption Landscape (Frequency fixed at {fixed_frequency:.2f})")
        plt.legend()
        plt.show()
