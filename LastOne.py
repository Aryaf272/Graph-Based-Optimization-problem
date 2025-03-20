import random
import time
import statistics
import matplotlib.pyplot as plt

# Node weights and edge connections

# Node weights and edge connections
node_values = {
    'A': 7,
    'B': 8,
    'C': 6,
    'D': 5,
    'E': 9,
    'F': 4
}

edge_values = {
    ('A', 'B'): 12,
    ('A', 'C'): 15,
    ('B', 'E'): 14,
    ('C', 'F'): 10,
    ('D', 'F'): 8,
    ('E', 'F'): 7,
    ('B', 'F'): 5,
    ('C', 'E'): 16,
    ('D', 'E'): 9
}

min_weight = 5
max_weight = 20
def compute_total_score(clusters, edges):
    total_score = 0
    for cluster in clusters:
        for i, first_node in enumerate(cluster):
            for second_node in cluster[i + 1:]:
                total_score += edges.get((first_node, second_node), 0) + edges.get((second_node, first_node), 0)
    return total_score


def compute_cluster_weights(clusters, node_values):
    return [sum(node_values[node] for node in cluster) for cluster in clusters]

def is_connected(node1, node2, edges):
    return (node1, node2) in edges or (node2, node1) in edges

def greedy_partitioning(node_values, min_weight, max_weight, edges):
    clusters = [[], []]
    cluster_weights = [0, 0]
    sorted_items = sorted(node_values.items(), key=lambda item: item[1], reverse=True)

    for node, weight in sorted_items:
        can_add_to_cluster = False

        for i in range(2):
            if cluster_weights[i] + weight <= max_weight:
                if len(clusters[i]) == 0 or any(is_connected(node, n, edges) for n in clusters[i]):
                    clusters[i].append(node)
                    cluster_weights[i] += weight
                    can_add_to_cluster = True
                    break

        if not can_add_to_cluster:
            print(f"Could not add node {node} to any cluster.")

    return clusters

def crossover(parent1, parent2):
    child = [[], []]
    used_nodes = set()

    for i in range(2):
        half_size = len(parent1[i]) // 2
        child[i] = parent1[i][:half_size] + parent2[i][half_size:]

        # Ensure no duplicates
        used_nodes.update(child[i])

    # Ensure clusters have unique nodes
    child[0] = list(set(child[0]))
    child[1] = list(set(child[1]))

    return child

def mutate(clusters, mutation_rate):
    for i in range(2):
        if random.random() < mutation_rate:
            if clusters[i]:
                node = random.choice(clusters[i])
                clusters[i].remove(node)

                # Only add the node to the other cluster if it doesn't already contain it
                if node not in clusters[1 - i]:
                    clusters[1 - i].append(node)
def genetic_algorithm(initial_population, edges, node_values, min_weight, max_weight, generations=100, mutation_rate=0.1):
    population = initial_population
    for generation in range(generations):
        population = sorted(population, key=lambda clusters: compute_total_score(clusters, edges), reverse=True)
        next_generation = population[:2]  # Keep the best 2

        while len(next_generation) < len(population):
            parent1, parent2 = random.choices(population[:10], k=2)  # Select parents
            child = crossover(parent1, parent2)
            mutate(child, mutation_rate)

            if is_valid(child, node_values, min_weight, max_weight):
                next_generation.append(child)

        population = next_generation

    return max(population, key=lambda clusters: compute_total_score(clusters, edges))
def is_valid(clusters, node_values, min_weight, max_weight):
    seen_nodes = set()
    for cluster in clusters:
        for node in cluster:
            if node in seen_nodes:
                return False  # Node is duplicated
            seen_nodes.add(node)
    weights = compute_cluster_weights(clusters, node_values)
    return all(min_weight <= weight <= max_weight for weight in weights)


#local search (hill) 
def search_for_improvement(initial_clusters, node_values, edges, min_weight, max_weight, iterations=10):
    best_clusters = initial_clusters
    best_score = compute_total_score(best_clusters, edges)

    for _ in range(iterations):
        new_clusters = swap_nodes(best_clusters, node_values, min_weight, max_weight)
        new_score = compute_total_score(new_clusters, edges)

        if new_score > best_score:
            best_score = new_score
            best_clusters = new_clusters

    return best_clusters, best_score

def swap_nodes(clusters, node_values, min_weight, max_weight):
    new_clusters = [cluster.copy() for cluster in clusters]
    first_cluster = random.choice(new_clusters)
    second_cluster = random.choice(new_clusters)

    if first_cluster and second_cluster and first_cluster != second_cluster:
        node_from_first = random.choice(first_cluster)
        node_from_second = random.choice(second_cluster)

        first_cluster.remove(node_from_first)
        second_cluster.remove(node_from_second)

        # Only add the nodes if they're not already in the other cluster
        if node_from_second not in first_cluster:
            first_cluster.append(node_from_second)
        if node_from_first not in second_cluster:
            second_cluster.append(node_from_first)

    weights = compute_cluster_weights(new_clusters, node_values)
    if all(min_weight <= weight <= max_weight for weight in weights):
        return new_clusters
    else:
        return clusters

def execute_multiple_searches(node_values, edges, min_weight, max_weight, iterations=10, trials=10):
    initial_clusters = greedy_partitioning(node_values, min_weight, max_weight, edges)
    initial_weights = compute_cluster_weights(initial_clusters, node_values)
    initial_score = compute_total_score(initial_clusters, edges)

    print(f"Initial Clusters (Greedy): {initial_clusters}, Weights: {initial_weights}, Score: {initial_score}\n")

    for i, cluster in enumerate(initial_clusters):
        print(f"Cluster {i + 1}: {cluster}")

    best_scores_greedy = []
    best_times_greedy = []
    best_scores_genetic = []
    best_times_genetic = []

    for trial in range(trials):
        # Hill Climbing
        start_time = time.time()
        improved_clusters, improved_score = search_for_improvement(initial_clusters, node_values, edges, min_weight, max_weight, iterations)
        end_time = time.time()

        elapsed_time = end_time - start_time

        print(f"Hill Climbing {trial + 1} Results: Clusters: {improved_clusters}, Weights: {compute_cluster_weights(improved_clusters, node_values)}, Score: {improved_score}, Duration: {elapsed_time:.4f} seconds")
        best_scores_greedy.append(improved_score)
        best_times_greedy.append(elapsed_time)

        # Genetic Algorithm
        start_time = time.time()
        initial_population = [initial_clusters] * 10  # Create a population of 10 identical clusters for GA
        best_clusters = genetic_algorithm(initial_population, edges, node_values, min_weight, max_weight)
        end_time = time.time()

        elapsed_time = end_time - start_time
        best_score = compute_total_score(best_clusters, edges)
        best_weights = compute_cluster_weights(best_clusters, node_values)

        print(f"Genetic Algorithm Trial {trial + 1} Results: Clusters: {best_clusters}, Weights: {best_weights}, Score: {best_score}, Duration: {elapsed_time:.4f} seconds\n")
        best_scores_genetic.append(best_score)
        best_times_genetic.append(elapsed_time)

    # Final Summary for Hill Climbing
    if best_scores_greedy:  # Ensure there are scores to calculate statistics
        best_score_greedy = max(best_scores_greedy)
        average_score_greedy = statistics.mean(best_scores_greedy)
        std_dev_greedy = statistics.stdev(best_scores_greedy)
        average_duration_greedy = statistics.mean(best_times_greedy)

        print(f"Final Summary for Hill Climbing:")
        print(f"Best Score (Hill Climbing): {best_score_greedy}")
        print(f"Average Score (Hill Climbing): {average_score_greedy:.2f}")
        print(f"Standard Deviation of Scores (Hill Climbing): {std_dev_greedy:.2f}")
        print(f"Average Duration (Hill Climbing): {average_duration_greedy:.4f} seconds\n")

    # Final Summary for Genetic Algorithm
    if best_scores_genetic:  # Ensure there are scores to calculate statistics
        best_score_genetic = max(best_scores_genetic)
        average_score_genetic = statistics.mean(best_scores_genetic)
        std_dev_genetic = statistics.stdev(best_scores_genetic)
        average_duration_genetic = statistics.mean(best_times_genetic)

        print(f"Final Summary for Genetic Algorithm:")
        print(f"Best Score (Genetic): {best_score_genetic}")
        print(f"Average Score (Genetic): {average_score_genetic:.2f}")
        print(f"Standard Deviation of Scores (Genetic): {std_dev_genetic:.2f}")
        print(f"Average Duration (Genetic): {average_duration_genetic:.4f} seconds\n")

    # Plotting the results
    # Plotting the results
    trials_range = range(1, trials + 1)

    plt.figure(figsize=(12, 6))

    # Plot Scores
    plt.subplot(1, 2, 1)
    plt.plot(trials_range, best_scores_greedy, marker='o', linestyle='-', color='pink', label='Hill Climbing')
    plt.plot(trials_range, best_scores_genetic, marker='x', linestyle='-', color='purple', label='Genetic Algorithm')
    plt.title('Scores per Trial')
    plt.xlabel('Trial Number')
    plt.ylabel('Score')
    plt.legend()
    plt.grid(True)

    # Plot Computation Times
    plt.subplot(1, 2, 2)
    plt.plot(trials_range, best_times_greedy, marker='o', linestyle='-', color='pink', label='Hill Climbing')
    plt.plot(trials_range, best_times_genetic, marker='x', linestyle='-', color='purple', label='Genetic Algorithm')
    plt.title('Computation Time per Trial')
    plt.xlabel('Trial Number')
    plt.ylabel('Time (seconds)')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()


# Run the experiments
execute_multiple_searches(node_values, edge_values, min_weight, max_weight, iterations=10, trials=10)
