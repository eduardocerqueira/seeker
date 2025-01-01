#date: 2025-01-01T16:28:09Z
#url: https://api.github.com/gists/d60c150c4ed7c2b12233131b88b89c46
#owner: https://api.github.com/users/PieroPaialungaAI

import pygad
import numpy
# Define the function to minimize
def func_to_minimize(x, y):
    return x * np.sin(4 * x) +1.1 * y * np.sin(2 * y)

# Define the fitness function for the GA
def fitness_func(ga_instance, solution, solution_idx):
    x, y = solution
    output = func_to_minimize(x, y)
    fitness = -output   # Adding a small constant to avoid division by zero
    return fitness

# Set up the GA parameters
num_generations = 400
num_parents_mating = 4
sol_per_pop = 8
num_genes = 2  # We have two variables: x and y

# The range for the initial population
init_range_low = 0
init_range_high = 10

parent_selection_type = "sss"
keep_parents = 1
crossover_type = "single_point"
mutation_type = "random"
mutation_percent_genes = 10

# Initialize the GA
ga_instance = pygad.GA(num_generations=num_generations,
                       num_parents_mating=num_parents_mating,
                       fitness_func=fitness_func,
                       sol_per_pop=sol_per_pop,
                       num_genes=num_genes,
                       init_range_low=init_range_low,
                       init_range_high=init_range_high,
                       parent_selection_type=parent_selection_type,
                       keep_parents=keep_parents,
                       crossover_type=crossover_type,
                       mutation_type=mutation_type,
                       mutation_percent_genes=mutation_percent_genes,
                      save_solutions=True)

# Run the GA
ga_instance.run()

# Get the best solution
solution, solution_fitness, solution_idx = ga_instance.best_solution()
print("Parameters of the best solution : {solution}".format(solution=solution))
print("Fitness value of the best solution = {solution_fitness}".format(solution_fitness=solution_fitness))

# Calculate the output using the best solution
predicted_output = func_to_minimize(solution[0], solution[1])
print("Predicted output based on the best solution : {prediction}".format(prediction=predicted_output))