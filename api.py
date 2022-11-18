# TODO:
# - comparisons between AIs
# - saving the model and weights seperately (and an API to use the AI)
# - save AI instances from pygad.GA

# available commands: train (create/load), play, test (test AI on all wordle words), plot (AI training data)


import json
from math import log2
import os
from os.path import join
import pygad
import tensorflow as tf
import matplotlib
import numpy

from general import *


def train():
    print("\nTraining selected.")
    
    training_type = None
    while (training_type not in ["create", "load"]):
        training_type = input("Create a new AI or load one? [create/load]: ")
    
    generations = int(input("Number of training generations: "))
    if (generations >= 2):
        create(generations)
    else:
        print("The number of training generations needs to be greater than one")
    
    if (training_type == "create"):
        from create import create
        create(generations)
    elif (training_type == "load"):
        from load import load
        load(generations)


def get_best_solution(solutions, model, data_outputs):
    def modified_fitness_func(solution):
        data_inputs = []

        fitness = 0.0
        
        remaining_words = wordle_words

        for _ in range(6):
            data = np.array(
                [data_inputs + [0.0] * (50 - len(data_inputs))])

            predictions = list(np.floor(predict(
                model=model, solution=solution, data=data)[0] * 26.0))

            colours = list(np.array(colour(predictions, data_outputs)) / 2)
            
            data_inputs.extend(predictions + colours)
            
            total_words = len(remaining_words)
            remaining_words = options_from_guess(remaining_words, colours, predictions)
            if len(remaining_words) == 0:
                return 0
            fitness += -log2(len(remaining_words) / total_words)

        return fitness
    
    words = [random_word() for _ in range(10)]
    
    best_solution = None
    best_fitness = 0
    for s in solutions:
        fitness = 0
        for i in range(10):
            fitness += modified_fitness_func(s, model, words[i])
        if (fitness > best_fitness):
            best_solution = s
            best_fitness = fitness
            
    return best_solution


def test():
    name = input("AI instance to test: ")
    model: tf.keras.Model = tf.keras.models.load_model(join("instances", name, "model"))
    ga_instance: pygad.GA = pygad.load(join("instances", name, "algorithm"))
    solution = get_best_solution(ga_instance.solutions, model)
    
    def modified_fitness_func(data_outputs):
        data_inputs = []

        fitness = 0.0
        turns = 1
        
        remaining_words = wordle_words

        for _ in range(6):
            data = np.array(
                [data_inputs + [0.0] * (50 - len(data_inputs))])

            predictions = list(np.floor(predict(
                model=model, solution=solution, data=data)[0] * 26.0))

            colours = list(np.array(colour(predictions, data_outputs)) / 2)
            
            data_inputs.extend(predictions + colours)
            
            total_words = len(remaining_words)
            remaining_words = options_from_guess(remaining_words, colours, predictions)
            if len(remaining_words) == 0:
                return 0
            fitness += -log2(len(remaining_words) / total_words)
            if len(remaining_words) == 1:
                return (fitness, turns)
            turns += 1

        return (fitness, 0) # 0 means did not finish

    # results[0] is the fitness values, results[1] is the turn in which it finished
    results = zip(*[modified_fitness_func(w) for w in wordle_words])
    # TODO: create scatter plot of data

# TODO: play function

def plot():
    matplotlib.pyplot.title(input("Figure title: "), fontsize=14)
    matplotlib.pyplot.xlabel(input("X label: "), fontsize=14)
    matplotlib.pyplot.ylabel(input("Y label: "), fontsize=14)
    
    colour_values = [
        "FF0000", "00FF00", "0000FF", "FFFF00", "FF00FF", "00FFFF", "000000", 
        "800000", "008000", "000080", "808000", "800080", "008080", "808080", 
        "C00000", "00C000", "0000C0", "C0C000", "C000C0", "00C0C0", "C0C0C0", 
        "400000", "004000", "000040", "404000", "400040", "004040", "404040", 
        "200000", "002000", "000020", "202000", "200020", "002020", "202020", 
        "600000", "006000", "000060", "606000", "600060", "006060", "606060", 
        "A00000", "00A000", "0000A0", "A0A000", "A000A0", "00A0A0", "A0A0A0", 
        "E00000", "00E000", "0000E0", "E0E000", "E000E0", "00E0E0", "E0E0E0", 
    ];
    
    for i in range(int(input("Number of AIs to plot: "))):
        name = input(f"Name {i+1}: ")
        with open(join("instances", name, "best_fitness_scores.txt"), "r") as file:
            jl = json.loads(file.read())
            best_fitness_scores: list = jl["best_fitness_scores"]
            indices = range(len(best_fitness_scores))
            
            matplotlib.pyplot.plot(best_fitness_scores, linewidth=3, color=colour_values[i])
            
            # trendline
            z = numpy.polyfit(indices, best_fitness_scores, 1)
            p = numpy.poly1d(z)
            matplotlib.pyplot.plot(p(indices), linewidth=3, color="#D2042D", alpha=0.5)
    
    os.makedirs("figures", exist_ok=True)
    save = None
    while (change_settings not in ["Y", "N"]):
        change_settings = input("Save figure? [Y/N]: ").capitalize()
    if save == "Y":
        matplotlib.pyplot.savefig(fname=input("Figure name: "),
                                    bbox_inches='tight')
    
    matplotlib.pyplot.show()


print("Welcome. Please select a mode to continue. Available modes: [train, play, test, plot]")
mode = input("Mode: ")
while (mode not in ["train", "play", "test", "plot"]):
    mode = input("Invalid input. Please try again.\nMode: ")

if (mode == "train"):
    train()
