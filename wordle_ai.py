import pygad
import pygad.kerasga
import input_set
import tensorflow as tf
import numpy as np


def init_organism():
    return [[], input_set.random_word()]


def fitness_func(solution, solution_idx):
    global model, organism_data
    
    [data_inputs, data_outputs] = organism_data[solution_idx]
    data_inputs += [0.0 for _ in range(50 - len(data_inputs))]
    data_inputs = np.array([data_inputs])
    
    predictions = np.floor(pygad.kerasga.predict(model=model, solution=solution, data=data_inputs)[0] * 26.0)
    
    colours = input_set.colour(list(predictions), data_outputs)
    
    if (len(organism_data[solution_idx][0]) < 50):
        organism_data[solution_idx][0].extend(list(np.floor(predictions)) + list(map(lambda x: x / 2, colours)))
    else:
        organism_data[solution_idx] = init_organism()

    fitness = colours.count(2) + 0.5 * colours.count(1)
    return fitness


# Keras model
layers = 10

model = tf.keras.models.Sequential([tf.keras.Input(shape=50)])
for _ in range(layers):
    model.add(tf.keras.layers.Dense(50, activation="sigmoid"))
model.add(tf.keras.layers.Dense(5, activation="sigmoid"))

num_solutions = 8
organism_data = [init_organism() for _ in range(num_solutions)]
keras_ga = pygad.kerasga.KerasGA(model=model, num_solutions=num_solutions)


# PyGAD parameters
num_generations = 100
num_parents_mating = 4
initial_population = keras_ga.population_weights

parent_selection_type = "rank"
crossover_type = "scattered"

keep_parents = 0
keep_elitism = 0

mutation_type = "random"
mutation_probability = 0.2

random_mutation_min_val = -1.0 / layers
random_mutation_max_val = 1.0 / layers

ga_instance = pygad.GA(num_generations=num_generations,
                       num_parents_mating=num_parents_mating,
                       initial_population=initial_population,
                       fitness_func=fitness_func,
                       parent_selection_type=parent_selection_type,
                       crossover_type=crossover_type,
                       keep_parents=keep_parents,
                       keep_elitism=keep_elitism,
                       mutation_type=mutation_type,
                       mutation_probability=mutation_probability,
                       random_mutation_min_val=random_mutation_min_val,
                       random_mutation_max_val=random_mutation_max_val)

ga_instance.run()
ga_instance.plot_fitness()

