import networkx as nx
import matplotlib.pyplot as plt
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
from itertools import combinations
from collections import defaultdict
import time
from numba import njit
import community as community

#@njit(parallel=True)
def opt_n_m_anonymity_adj(adj_matrix):
    n_m_anonymity_dict = defaultdict(list)
    
    num_nodes = adj_matrix.shape[0]

    for node in range(num_nodes):
        # Step 1: Determine ego state (neighbors and edges)
        neighbors = np.nonzero(adj_matrix[node])[0]
        num_neighbors = len(neighbors)
        
        # Use Numpy operations for efficiency
        neighborhood_edges = int(np.count_nonzero(adj_matrix[neighbors][:, neighbors])/2)

        ego_state = (num_neighbors + 1, neighborhood_edges+num_neighbors)

        # Step 2: Count unique equivalence classes
        n_m_anonymity_dict[ego_state].append(node)

    number_unique = sum(1 for equivalence_class in n_m_anonymity_dict.values() if len(equivalence_class) == 1)

    return n_m_anonymity_dict, number_unique

# Function to calculate n-m-anonymity
def calculate_n_m_anonymity(graph, deleted_edges, edges_to_delete):
    # Create a copy of the graph to avoid modifying the original graph
    modified_graph = graph.copy()

    # Delete edges based on the binary string
    for edge, delete_flag in zip(edges_to_delete, deleted_edges):
        if delete_flag:
            modified_graph.remove_edge(*edge)

    # Calculate n-m-anonymity
    modified_graph_adj = nx.to_numpy_array(modified_graph).astype(int)
    _, fitness = opt_n_m_anonymity_adj(modified_graph_adj)

    return fitness

def unique_edges_calculator(graph, deleted_edges, edges_to_delete):
    modified_graph = graph.copy()

    # Delete edges based on the binary string
    for edge, delete_flag in zip(edges_to_delete, deleted_edges):
        if delete_flag:
            modified_graph.remove_edge(*edge)

    adj_G = nx.to_numpy_array(modified_graph).astype(int)
    n_m_anonymity_dict, number_unique = opt_n_m_anonymity_adj(adj_G)
    
    node_names = []
    for nodes in modified_graph.nodes():
        node_names.append(nodes)
    
    unique_nodes = []
    for equivalence_class in n_m_anonymity_dict.values():
        if len(equivalence_class) == 1:
            unique_nodes.append(equivalence_class[0])
    
    unique_edges = []
    for node in unique_nodes:
        # Step 1: Determine ego state (neighbors and edges)
        neighbors = np.nonzero(adj_G[node])[0]
        
        for neigh in neighbors:
            a = min(neigh, node)
            b = max(neigh, node)
            unique_edges.append((node_names[a], node_names[b]))
    
    
    unique_edges = list(dict.fromkeys(unique_edges))

    return unique_edges

# Function to perform mutation with 1% constraint
#@njit(parallel=True)
def mutate(graph, edges_to_delete, individual, allowance=0.01, mutation_rate = 0.05):
    
    allowed_edges = unique_edges_calculator(graph, individual, edges_to_delete)
    allowed_edges = set(allowed_edges)
    
    mutation_mask = np.random.rand(len(individual)) < mutation_rate
    
    for i, (mask, edge) in enumerate(zip(mutation_mask, graph.edges())):
        if mask == True:
            if individual[i] == 0:
                if edge not in allowed_edges:
                    mutation_mask[i] = False
                
    individual[mutation_mask] = 1 - individual[mutation_mask]
    
    return individual


#@njit(parallel=True)
def crossover(graph, edges_to_delete, parent1, parent2, n_points=2, allowance=0.01, mutation_rate=0.05):
    if n_points == 111:
        # Initialize child arrays
        child1 = np.zeros_like(parent1)
        child2 = np.zeros_like(parent2)

        # Perform uniform crossover
        for i in range(len(parent1)):
            # Flip a coin for each gene to decide the inheritance
            if np.random.rand() < 0.5:
                child1[i] = parent1[i]
                child2[i] = parent2[i]
            else:
                child1[i] = parent2[i]
                child2[i] = parent1[i]

        # Apply mutation to children to enforce 1% constraint
        child1 = mutate(graph, edges_to_delete, child1, allowance, mutation_rate)
        child2 = mutate(graph, edges_to_delete, child2, allowance, mutation_rate)

    else:
        # Generate n crossover points
        crossover_points = np.sort(np.random.choice(len(parent1), n_points, replace=False))

        # Initialize child arrays
        child1 = np.zeros_like(parent1)
        child2 = np.zeros_like(parent2)

        # Perform crossover
        for i in range(n_points + 1):
            start = 0 if i == 0 else crossover_points[i - 1] + 1
            end = len(parent1) if i == n_points else crossover_points[i]

            # Alternate parents for each segment
            if i % 2 == 0:
                child1[start:end] = parent1[start:end]
                child2[start:end] = parent2[start:end]
            else:
                child1[start:end] = parent2[start:end]
                child2[start:end] = parent1[start:end]

        # Apply mutation to children to enforce 1% constraint
        child1 = mutate(graph, edges_to_delete, child1, allowance, mutation_rate)
        child2 = mutate(graph, edges_to_delete, child2, allowance, mutation_rate)

    return child1, child2


def roulette_wheel_selection(fitness_values, num_parents=100):
    # Calculate the inverted fitness values (subtract from the maximum fitness)
    inverted_fitness = np.nan_to_num((max(fitness_values)+1) - fitness_values)
    
    # Calculate the total inverted fitness
    total_inverted_fitness = np.sum(inverted_fitness)

    # Check if total_inverted_fitness is zero to avoid division by zero
    if total_inverted_fitness == 0:
        # Handle the case where all fitness values are the same or NaN
        return np.random.choice(len(fitness_values), num_parents)
    
    else:
        # Calculate selection probabilities
        selection_probabilities = inverted_fitness / total_inverted_fitness

        # Use np.random.choice for roulette wheel selection
        parents_indices = np.random.choice(len(fitness_values), num_parents, p=selection_probabilities)

        return parents_indices

def fitness_calculator(popn, uniqueness, allowance):
    fitness = []
    penalty = []
    
    for i, indiv in enumerate(popn):
        current_num_edges_to_set_one = sum(indiv)
        remaining_num_edges_to_set_one = current_num_edges_to_set_one -  max(1, int(allowance * len(indiv))) 

        if remaining_num_edges_to_set_one > 0:
            penalty.append(remaining_num_edges_to_set_one)
        else:
            penalty.append(0)

        fitness.append(uniqueness[i] + penalty[i])

    return fitness


def tournament_selection(fitness_values, tournament_size=5, num_parents=100):
    num_individuals = len(fitness_values)
    parents_indices = []

    for _ in range(num_parents):
        # Randomly select individuals for the tournament
        tournament_indices = np.random.choice(num_individuals, tournament_size, replace=False)
        
        # Find the index of the individual with the highest fitness in the tournament
        winner_index = tournament_indices[np.argmin(fitness_values[tournament_indices])]
        
        # Add the winner's index to the list of selected parents
        parents_indices.append(winner_index)

    return parents_indices


def main(dataset, popn_init, p_selection, n_points, m_rate, e_selection, number_runs, rd_number = 0, config_number = 1, iteration = 0):
    if dataset == "CA-GrQc":
        mediumG = nx.read_edgelist(r"./../../CA-GrQc.txt")
    elif dataset == "CollegeMsg":
        mediumG = nx.read_edgelist(r"./../../CollegeMsg_noTime.txt")
    elif dataset == "Blogs":
        mediumG = nx.read_edgelist(r"./../../moreno_blogs.txt")

    mediumG.remove_edges_from(nx.selfloop_edges(mediumG))
    
    adj_G = nx.to_numpy_array(mediumG).astype(int)

    num_edges = len(mediumG.edges)
    allowance = 0.05
    #n_points = 100
    population_size = 100
    #generations = 1000
    
    if m_rate == 111:
        mutation_rate = 0.0005
        eta = 0.000025
    elif m_rate == 1111:
        mutation_rate = 0.0005
        eta = 0.00001
    else:
        mutation_rate = 0.0001
        eta = 0

    saved_fit = []

    if rd_number == 0:
        # Initialize the population with at least 1% of edges set to 1
        population = np.zeros((population_size, num_edges), dtype=int)

        #allowance_num = int(allowance*num_edges)
        for i in range(population_size):
            num_edges_to_set_one = max(1, int(popn_init * num_edges))
            edges_to_set_one = np.random.choice(num_edges, num_edges_to_set_one, replace=False)
            population[i, edges_to_set_one] = 1
    else:
        popn_file_path = "./{}/rd{}/population_config{}.npy".format(dataset, rd_number-1, config_number)
        population = np.load(popn_file_path)


    #population = np.load(f'population_0.05_100_comm.npy')

    count = 0
    # Main loop for genetic algorithm
    for generation in range(number_runs):
        #print(count)
        count+=1

        if count == 1:
            # Evaluate the fitness of each individual
            #fitness_values = np.array([calculate_n_m_anonymity_sorted(mediumG, sorted_edges, individual) for individual in population])
            
            unique_values = np.array([calculate_n_m_anonymity(mediumG, individual, mediumG.edges()) for individual in population])
            fitness_values = np.array(fitness_calculator(population, unique_values, allowance))

        print("{},{},{},{}, {}".format(rd_number, config_number, iteration, count, np.min(fitness_values)))
        saved_fit.append(np.min(fitness_values))
        
        # Generate parents
        if p_selection == 1:
            parents_indices = roulette_wheel_selection(fitness_values, num_parents = 150)
        elif p_selection == 2:
            parents_indices = tournament_selection(fitness_values, tournament_size = 5, num_parents = 150)

        # Create the next generation through crossover and mutation
        children = []
        
        mutation_rate = max(1/num_edges, mutation_rate * (1 - eta * count))

        # Create the next generation through crossover and mutation
        children = []
        for i in range(0, len(parents_indices)-1, 2):  # Ensure that the loop doesn't go out of bounds
            parent1, parent2 = population[parents_indices[i]], population[parents_indices[i + 1]]
            child1, child2 = crossover(mediumG, mediumG.edges(), parent1, parent2, n_points, allowance, mutation_rate)
            children.extend([child1, child2])

         # Evaluate fitness values of children
        unique_values_children = np.array([calculate_n_m_anonymity(mediumG, child, mediumG.edges()) for child in children])
        fitness_values_children = np.array(fitness_calculator(children, unique_values_children, allowance))

        parents_children = np.concatenate((population, children))
        parents_children_fitness = np.hstack((fitness_values, fitness_values_children))
        parents_children_unique = np.hstack((unique_values, unique_values_children))

        sorted_parents_children = np.argsort(parents_children_fitness)
        parents_children = parents_children[sorted_parents_children]
        parents_children_fitness = parents_children_fitness[sorted_parents_children]
        parents_children_unique = parents_children_unique[sorted_parents_children]


        if e_selection == 1:
            new_parents_indices = roulette_wheel_selection(sorted_parents_children, num_parents = 100)
            population = parents_children[new_parents_indices]
            fitness_values = parents_children_fitness[new_parents_indices]
            unique_values = parents_children_unique[new_parents_indices]
        elif e_selection == 2:
            new_parents_indices = tournament_selection(sorted_parents_children, 5, num_parents = 100)
            population = parents_children[new_parents_indices]
            fitness_values = parents_children_fitness[new_parents_indices]
            unique_values = parents_children_unique[new_parents_indices]
        else:
            population = parents_children[0:population_size]
            fitness_values = parents_children_fitness[0:population_size]
            unique_values = parents_children_unique[0:population_size]

        # Check if the last 10 values are the same
        if len(saved_fit) >= 40 and all(saved_fit[-1] == value for value in saved_fit[-40:]):
            break

    np.save(f"./FINALRD/{dataset}/FINAL/population_config{config_number}_{iteration}", population)
    np.save(f"./FINALRD/{dataset}/FINAL/saved_fit_config{config_number}_{iteration}", saved_fit)
    #np.save(f"./{dataset}/rd{rd_number}/sortededges_config{config_number}", sorted_edges)

    #fitness_values = np.array([calculate_n_m_anonymity_sorted(mediumG, sorted_edges, individual) for individual in population])

    final_fitness = np.min(fitness_values)

    # print the result
    print("Final Fitness:", final_fitness)

    return final_fitness