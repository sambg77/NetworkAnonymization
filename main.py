import os
import itertools
import random
import numpy as np

from Basic_GA import main as Basic_main
from Mutation_GA import main as Mutation_main
from Sorting_GA import main as Sorting_main

selected_configs = [(0.005, 2, 111, 0.0001, 3)]

dataset = "CA-GrQc"

configuration = "Sorting"

num_iterations = 5

# if configuration is sorting
sorting = "degree"

if configuration == "Sorting":
    # Define the file path
    file_path =  f"./New_Results/{configuration}/FINALRD/{sorting}/{dataset}"
    configuration_file_path = f"{file_path}/selected_configs.txt"

    # Create the directory if it doesn't exist
    os.makedirs(file_path, exist_ok=True)
else: 
    # Define the file path
    file_path =  f"./New_Results/{configuration}/FINALRD/{dataset}"
    configuration_file_path = f"{file_path}/selected_configs.txt"

    # Create the directory if it doesn't exist
    os.makedirs(file_path, exist_ok=True)



# Open the file in write mode
with open(configuration_file_path, "w") as file:
    # Write each configuration to the file
    for config in selected_configs:
        file.write(','.join(map(str, config)) + '\n')


for rd_number in range(len(selected_configs)):
    for i in range(num_iterations):
        if configuration == "Basic":  
            final_fitness = Basic_main(dataset, selected_configs[rd_number][0], selected_configs[rd_number][1], selected_configs[rd_number][2], selected_configs[rd_number][3], selected_configs[rd_number][4], 1000, rd_number = 0, config_number = rd_number, iteration = i) #USUALLY CONFIG_NUMBER = RD_NUMBER
        elif configuration == "Mutation":
            final_fitness = Mutation_main(dataset, selected_configs[rd_number][0], selected_configs[rd_number][1], selected_configs[rd_number][2], selected_configs[rd_number][3], selected_configs[rd_number][4], 1000, rd_number = 0, config_number = rd_number, iteration = i) #USUALLY CONFIG_NUMBER = RD_NUMBER
        elif configuration == "Sorting":
            final_fitness = Sorting_main(dataset, sorting, selected_configs[rd_number][0], selected_configs[rd_number][1], selected_configs[rd_number][2], selected_configs[rd_number][3], selected_configs[rd_number][4], 1000, rd_number = 0, config_number = rd_number, iteration = i)
