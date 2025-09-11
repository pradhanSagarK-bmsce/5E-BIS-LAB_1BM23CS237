import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.metrics import pairwise_distances_argmin


NUM_CLUSTERS = 3
NUM_PARTICLES = 30
MAX_ITER = 20
DIMENSIONS = 2
SEED = 42
np.random.seed(SEED)


iris = load_iris()
data = iris.data[:, :DIMENSIONS]

def fitness(centroids, data):
    assignments = pairwise_distances_argmin(data, centroids)
    cost = 0
    for i in range(len(centroids)):
        cluster_points = data[assignments == i]
        if len(cluster_points) > 0:
            cost += np.sum((cluster_points - centroids[i]) ** 2)
    return cost

def initialize_particles(data, n_clusters, n_particles):
    particles = []
    for _ in range(n_particles):
        idx = np.random.choice(len(data), n_clusters, replace=False)
        particles.append(data[idx].copy())
    return np.array(particles)

def update_velocity(velocities, positions, pbest_positions, gbest_position, w=0.5, c1=1.5, c2=1.5):
    r1 = np.random.rand(*velocities.shape)
    r2 = np.random.rand(*velocities.shape)
    cognitive = c1 * r1 * (pbest_positions - positions)
    social = c2 * r2 * (gbest_position - positions)
    return w * velocities + cognitive + social

def pso_clustering(data, n_clusters, n_particles, max_iter):
    particles = initialize_particles(data, n_clusters, n_particles)
    velocities = np.zeros_like(particles)

    pbest_positions = particles.copy()
    pbest_fitness = np.array([fitness(p, data) for p in particles])

    gbest_idx = np.argmin(pbest_fitness)
    gbest_position = pbest_positions[gbest_idx].copy()
    gbest_fitness = pbest_fitness[gbest_idx]

    for iter in range(1, max_iter + 1):
        velocities = update_velocity(velocities, particles, pbest_positions, gbest_position)
        particles += velocities

        for i in range(n_particles):
            fit = fitness(particles[i], data)
            if fit < pbest_fitness[i]:
                pbest_fitness[i] = fit
                pbest_positions[i] = particles[i].copy()
                if fit < gbest_fitness:
                    gbest_fitness = fit
                    gbest_position = particles[i].copy()

        print(f"Iteration {iter}/{max_iter} - Best Fitness: {gbest_fitness:.4f}")

    return gbest_position


plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.scatter(data[:, 0], data[:, 1], c='gray', s=30)
plt.title("Before Clustering (Iris Dataset)")
plt.xlabel(iris.feature_names[0])
plt.ylabel(iris.feature_names[1])


best_centroids = pso_clustering(data, NUM_CLUSTERS, NUM_PARTICLES, MAX_ITER)
assignments = pairwise_distances_argmin(data, best_centroids)

clusters = {
    1 : 'versicolor',
    2 : 'virginica',
    3 : 'setosa'
}

plt.subplot(1, 2, 2)
colors = ['green', 'yellow', 'purple']
for i in range(NUM_CLUSTERS):
    cluster_points = data[assignments == i]
    plt.scatter(cluster_points[:, 0], cluster_points[:, 1], c=colors[i], s=30, label=f'{clusters[i+1]}')
plt.scatter(best_centroids[:, 0], best_centroids[:, 1], c='black', marker='x', s=150, label='Centroids')
plt.title("After Clustering (PSO)")
plt.xlabel(iris.feature_names[0])
plt.ylabel(iris.feature_names[1])
plt.legend()
plt.tight_layout()
plt.show()
