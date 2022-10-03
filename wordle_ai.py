import pygad
import fitness

num_generations = 100
num_parents_mating = 4

fitness_func = fitness.fitness_func

sol_per_pop = fitness.ORGANISM_COUNT
# total number of weights = number of nodes * number of weights per node
num_genes = (fitness.N + 1) * fitness.NODE_COUNT

init_range_low = -1.0 / fitness.L
init_range_high = 1.0 / fitness.L

parent_selection_type = "rank"

crossover_type = "scattered"

mutation_type = "random"
mutation_probability = 0.1

random_mutation_min_val = -1.0 / fitness.L
random_mutation_max_val = 1.0 / fitness.L

ga_instance = pygad.GA(num_generations=num_generations,
                       num_parents_mating=num_parents_mating,
                       fitness_func=fitness_func,
                       sol_per_pop=sol_per_pop,
                       num_genes=num_genes,
                       init_range_low=init_range_low,
                       init_range_high=init_range_high,
                       parent_selection_type=parent_selection_type,
                       crossover_type=crossover_type,
                       mutation_type=mutation_type,
                       mutation_probability=mutation_probability,
                       random_mutation_min_val=random_mutation_min_val,
                       random_mutation_max_val=random_mutation_max_val)

ga_instance.run()

ga_instance.plot_fitness()
