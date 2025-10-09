import random
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from itertools import permutations

class AntColony:
    def __init__(self, network, num_ants, num_iterations, alpha, beta, rho, Q):
        self.network = network  # Graph of the network
        self.num_nodes = len(network)
        self.num_ants = num_ants
        self.num_iterations = num_iterations
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.Q = Q

        # Initialize pheromone matrix
        self.pheromone = np.ones((self.num_nodes, self.num_nodes)) * 0.1
        self.best_length = float('inf')
        self.best_path = []

    def compute_total_distance(self, path):
        # Calculate total distance (or cost) of a given path
        total_cost = 0
        for i in range(len(path) - 1):
            cost = self.network[path[i]][path[i + 1]]
            if cost == 0 and path[i] != path[i+1]: 
                return float('inf')
            total_cost += cost
        return total_cost

    def choose_next_node(self, current_node, visited):
        # Calculate the probability of visiting each unvisited node
        probabilities = []
        possible_next_nodes = []
        total_weight = 0 # Use total_weight for the sum of pheromone * heuristic

        for next_node in range(self.num_nodes):
            if next_node not in visited and self.network[current_node][next_node] != 0:
                pheromone = self.pheromone[current_node][next_node] ** self.alpha
                heuristic = (1 / self.network[current_node][next_node]) ** self.beta if self.network[current_node][next_node] != 0 else 0 
                weight = pheromone * heuristic
                probabilities.append(weight)
                total_weight += weight
                possible_next_nodes.append(next_node)

        if total_weight == 0:
            return -1

        # Normalize probabilities only for possible next nodes
        normalized_probabilities = [prob / total_weight for prob in probabilities]
        return int(np.random.choice(possible_next_nodes, p=normalized_probabilities))

    def reroute_packet(self, source, destination):
        best_path_this_iter = []
        best_length_this_iter = float('inf')

        # Ant-based search
        for _ in range(self.num_ants):
            visited = {source}
            path = [source]

            current_node = source
            while current_node != destination and len(visited) < self.num_nodes:
                next_node = self.choose_next_node(current_node, visited)
                if next_node == -1:  # No valid next move
                    break
                path.append(next_node)
                visited.add(next_node)
                current_node = next_node

            # If the ant reached the destination, calculate the path length
            if path[-1] == destination:
                path_length = 0
                valid_path = True
                for i in range(len(path) - 1):
                    cost = self.network[path[i]][path[i + 1]]
                    if cost == 0:  # Should not happen with choose_next_node, but as a safeguard
                        valid_path = False
                        break
                    path_length += cost

                if valid_path and path_length < best_length_this_iter:
                    best_length_this_iter = path_length
                    best_path_this_iter = path

        # Update pheromone levels based on the best path found in this iteration
        if best_length_this_iter != float('inf'):
            for i in range(len(best_path_this_iter) - 1):
                self.pheromone[best_path_this_iter[i]][best_path_this_iter[i + 1]] += self.Q / best_length_this_iter
                self.pheromone[best_path_this_iter[i + 1]][best_path_this_iter[i]] += self.Q / best_length_this_iter

        # Evaporate pheromones
        self.pheromone *= (1 - self.rho)

        return best_path_this_iter, best_length_this_iter

    def run(self, source, destination):
        self.best_length = float('inf')
        self.best_path = []

        for iteration in range(self.num_iterations):
            path, length = self.reroute_packet(source, destination)
            if length < self.best_length:
                self.best_length = length
                self.best_path = path

        return self.best_path, self.best_length


# Function to calculate the total cost of a given TSP path
def tsp(cost):
    numNodes = len(cost)
    nodes = list(range(1, numNodes))  # Nodes excluding the starting point (0)

    minCost = float('inf')
    bestPath = []

    # Generate all permutations of the remaining nodes
    for perm in permutations(nodes):
        currCost = 0
        currNode = 0

        # Calculate the cost of the current permutation
        for node in perm:
            currCost += cost[currNode][node]
            currNode = node

        # Add the cost to return to the starting node (0)
        currCost += cost[currNode][0]

        # Update the minimum cost if the current cost is lower
        if currCost < minCost:
            minCost = currCost
            bestPath = [0] + list(perm) + [0]  # Adding the start point (0) at the beginning and end

    return bestPath, minCost


# Visualization of the TSP problem
def visualize_tsp(cost, best_path, cities):
    numCities = len(cost)
    
    # Create a graph to represent the cities and edges
    G = nx.Graph()
    
    # Add edges to the graph based on the cost matrix
    for i in range(numCities):
        for j in range(i + 1, numCities):
            if cost[i][j] != 0:
                G.add_edge(i, j, weight=cost[i][j])

    # Positioning nodes using a spring layout (or you can use a circular layout)
    pos = nx.spring_layout(G, seed=42)  # You can also use nx.circular_layout(G)

    # Plot the graph
    plt.figure(figsize=(8, 6))
    nx.draw(G, pos, with_labels=True, node_size=700, node_color='skyblue', font_weight='bold', font_size=15, labels={i: cities[i] for i in range(len(cities))})
    
    # Add edge weights (costs)
    labels = nx.get_edge_attributes(G, 'weight')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)

    # Highlight the TSP path
    path_edges = []
    for i in range(len(best_path) - 1):
        path_edges.append((best_path[i], best_path[i + 1]))
    
    # Draw the best path in red
    nx.draw_networkx_edges(G, pos, edgelist=path_edges, edge_color='red', width=3)

    plt.title("Traveling Salesman Problem - Shortest Path")
    plt.show()


if __name__ == "__main__":

    # 1st Problem: Ant Colony Optimization (for routing)

    # Example network graph (adjacency matrix)
    network = [
        [0, 2, 2, 5, 0],
        [2, 0, 3, 8, 2],
        [2, 3, 0, 1, 1],
        [5, 8, 1, 0, 3],
        [0, 2, 1, 3, 0]
    ]

    # Ant Colony Parameters
    num_ants = 10
    num_iterations = 100
    alpha = 1       # Pheromone importance
    beta = 2        # Heuristic importance (1 / cost)
    rho = 0.1       # Pheromone evaporation rate
    Q = 100         # Pheromone deposit constant

    # Initialize Ant Colony
    ac = AntColony(network, num_ants, num_iterations, alpha, beta, rho, Q)

    source = 0
    destination = 3

    best_path, best_length = ac.run(source, destination)

    print("Ant Colony Optimization (Shortest Path):")
    print("Best Path:", best_path)
    print("Best Path Length:", best_length)

    # 2nd Problem: Traveling Salesman Problem (TSP)

    # City names around Bangalore
    cities = ["Bangalore (BLR)", "Mysuru (MYS)", "Channarayapatna (CHP)", "Tumkur (TUM)", "Hosur (HSR)"]

    # Example cost matrix for cities around Bangalore
    cost = [
        [0, 150, 120, 80, 160],  # Bangalore to other cities
        [150, 0, 80, 70, 190],   # Mysuru to other cities
        [120, 80, 0, 90, 130],   # Channarayapatna to other cities
        [80, 70, 90, 0, 110],    # Tumkur to other cities
        [160, 190, 130, 110, 0]  # Hosur to other cities
    ]

    # Solve TSP
    best_path_tsp, min_cost = tsp(cost)

    print("\nTraveling Salesman Problem (TSP):")
    print("Best Path:", [cities[i] for i in best_path_tsp])  
    print("Minimum Cost:", min_cost)

    # Visualize TSP path
    visualize_tsp(cost, best_path_tsp, cities)
    
    # Visualization of Ant Colony Network and Best Path
    G = nx.Graph()
    for i in range(len(network)):
        for j in range(i + 1, len(network)):
            if network[i][j] != 0:
                G.add_edge(i, j, weight=network[i][j])

    pos = nx.spring_layout(G)
    plt.figure(figsize=(8, 6))
    nx.draw(G, pos, with_labels=True, node_size=700, node_color='skyblue', font_weight='bold', font_size=15)
    labels = nx.get_edge_attributes(G, 'weight')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)

    # Highlight the best path from Ant Colony
    path_edges = []
    for i in range(len(best_path) - 1):
        if G.has_edge(best_path[i], best_path[i+1]):
            path_edges.append((best_path[i], best_path[i+1]))
        elif G.has_edge(best_path[i+1], best_path[i]):
            path_edges.append((best_path[i+1], best_path[i]))

    nx.draw_networkx_edges(G, pos, edgelist=path_edges, edge_color='red', width=3)
    plt.title("Ant Colony Optimization - Best Path")
    plt.show()
