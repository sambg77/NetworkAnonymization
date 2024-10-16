# Network Anonymization as an Optimization Problem

## Abstract

Publishing sensitive data, including social network data, poses a significant risk of breaching user privacy. Traditional network anonymization methods often focus on fully anonymizing datasets. Despite efforts to minimize alterations, these methods result in substantial utility loss in the anonymized network. This work proposes a novel approach to network anonymity by framing it as an optimization problem, aiming to balance anonymity and utility more effectively. We introduce three variations of a genetic algorithm (GA) to address this optimization problem, two of which utilize domain-specific knowledge to exploit the underlying structure of networks. This approach minimizes the number of unique nodes while allowing only a 5% deletion of edges, resulting in high network utility. The proposed algorithms were tested on various real-world networks. Among them, the GA with uniqueness-aware mutation consistently outperformed the others, anonymizing 72% of unique nodes while deleting only 1.6% of the edges. Additionally, we offer insights into the edges deleted by our algorithms, demonstrating that they consistently target edges connected to central nodes. Overall, this work demonstrates that framing network anonymity as an optimization problem can significantly improve utility preservation, offering a promising direction for future research.

## Dependencies

Use the conda to install all the required packages.

```bash
conda create --name <env> --file installed_packages.txt
```

## Usage
Update the following parameters in the main.py file:

```python
selected_configs = [(0.005, 2, 111, 0.0001, 3)]

dataset = "CA-GrQc"

configuration = "Sorting"

num_iterations = 5

# if configuration is sorting
sorting = "degree"
```

#### selected_configs

1st parameter: probability of flipping a 0 to 1 in initial population

2nd parameter: parental selection (1 = RW selection, 2 = TS)

3rd parameter: number of crossover points (111 = uniform crossover)

4th parameter: mutation rate (111 and 1111 are special mutation rates with a decay rate)

5th parameter: environmental selection (1 = RW selection, 2 = TS, 3 = elitist selection)

#### dataset
dataset must be set to one of the following {"CA-GrQc", "Blogs", "CollegeMsg"}.

#### configuration
configuration must be set to one of the following {"Basic", "Mutation", "Sorting"}.

#### num_iterations
This is the number of iterations for which you want to run each configuration.

#### sorting
This parameter determines how we sort the edges if we are using the "Sorting" configuration. Must be one of the following {"BC", "degree", "close", "comm"}.

