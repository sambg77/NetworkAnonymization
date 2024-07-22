import itertools
import random
import numpy as np

from GA_SH import main

popn_init_options = [0.005]
p_selection_options = [1,2]
n_points_options = [10,111]
m_rate_options = [1111]
e_selection_options = [3] 


#configs_setups = [50, 25, 13, 5, 2]
#number_runs = [10, 20, 40, 80, 1000]

all_combinations = list(itertools.product(popn_init_options, p_selection_options, n_points_options, m_rate_options, e_selection_options))
random.shuffle(all_combinations)

selected_configs = all_combinations

selected_configs = [(0.005, 2, 111, 0.0001, 3)]

#selected_configs[0] = (0.0025, 1, 111, 111, 3)

dataset = "Blogs"

# Define the file path
file_path = f"./FINALRD/{dataset}/FINAL/selected_configs1.txt"

# Open the file in write mode
with open(file_path, "w") as file:
    # Write each configuration to the file
    for config in selected_configs:
        file.write(','.join(map(str, config)) + '\n')


for rd_number in range(len(selected_configs)):
    for i in range(5):  
        final_fitness = main(dataset, selected_configs[rd_number][0], selected_configs[rd_number][1], selected_configs[rd_number][2], selected_configs[rd_number][3], selected_configs[rd_number][4], 1000, rd_number = 0, config_number = 3, iteration = i) #USUALLY CONFIG_NUMBER = RD_NUMBER

