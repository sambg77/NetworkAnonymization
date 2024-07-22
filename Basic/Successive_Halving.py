import itertools
import random
import numpy as np

from GA_SH import main

popn_init_options = [0.0025, 0.005, 0.02]
p_selection_options = [1,2]
n_points_options = [10,25,100,111]
m_rate_options = [0.0001, 111, 1111]
e_selection_options = [1,2,3] 


configs_setups = [50, 25, 13, 5, 2]
number_runs = [10, 20, 40, 80, 1000]

all_combinations = list(itertools.product(popn_init_options, p_selection_options, n_points_options, m_rate_options, e_selection_options))
random.shuffle(all_combinations)

selected_configs = all_combinations[:50]

#selected_configs[0] = (0.0025, 1, 111, 111, 3)

# Define the file path
file_path = "./Blogs/selected_configs.txt"

# Open the file in write mode
with open(file_path, "w") as file:
    # Write each configuration to the file
    for config in selected_configs:
        file.write(','.join(map(str, config)) + '\n')


for rd_number in range(5):
    configs_fitness = np.zeros(50, dtype=int)

    for config_number in range(configs_setups[rd_number]):
        if rd_number == 0:
            final_fitness = main(selected_configs[config_number][0], selected_configs[config_number][1], selected_configs[config_number][2], selected_configs[config_number][3], selected_configs[config_number][4], number_runs[rd_number], rd_number, config_number)

            configs_fitness[config_number] = final_fitness
        else:
            sorted_config_number = sorted_indices[config_number]
            final_fitness = main(selected_configs[sorted_config_number][0], selected_configs[sorted_config_number][1], selected_configs[sorted_config_number][2], selected_configs[sorted_config_number][3], selected_configs[sorted_config_number][4], number_runs[rd_number], rd_number, sorted_config_number)

            configs_fitness[sorted_config_number] = final_fitness
            
   # Convert configs_fitness to a numpy array
    configs_fitness = np.array(configs_fitness)

    # Get the indices of zeros
    zero_indices = np.where(configs_fitness == 0)[0]

    # Get the indices of nonzero values
    nonzero_indices = np.where(configs_fitness != 0)[0]
    nonzero_configs_fitness = configs_fitness[nonzero_indices]

    # Sort the indices based on minimizing final fitness for nonzero values
    sorted_nonzero_indices = nonzero_indices[np.argsort(nonzero_configs_fitness)]

    # Concatenate the sorted indices of nonzero values with the indices of zeros
    sorted_indices = np.concatenate((sorted_nonzero_indices, zero_indices))

    np.save("./Blogs/rd{}/sorted_indices".format(rd_number), sorted_indices)
    

