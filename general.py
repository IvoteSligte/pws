from collections import defaultdict
import copy
from math import floor
import time
import pygad
import pygad.kerasga
import os
from os.path import join
from settings import Settings
import numpy as np
import random
import json
from matplotlib import pyplot as plt


possible_wordle_words = [tuple((ord(l) - 97) for l in word.rstrip())
                for word in open("possible_words.txt")]


allowed_wordle_words = frozenset(tuple((ord(l) - 97) for l in word.rstrip()) for word in open("allowed_words.txt"))


def random_wordle_word():
    global possible_wordle_words
    random.seed()
    return possible_wordle_words[random.randrange(len(possible_wordle_words))]


num_generations = None
generations_passed = 0
fitness_scores_since_save = [[]]
first_guesses_since_save = [[]]
start_time = 0
last_save_time = 0
save_count = 0
ai_name = None
correct_output_values = [random_wordle_word() for _ in range(10)]

def on_start(ga_instance):
    global start_time, last_save_time
    start_time = time.time_ns()
    last_save_time = time.time_ns()


def format_secs(s):
    s = floor(s)
    m = floor(s / 60) % 60
    h = floor(floor(s / 60) / 60)
    s = s % 60

    return f"{h}:{m}:{s}"


def on_generation(ga_instance):
    global save_count, last_save_time, fitness_scores_since_save, first_guesses_since_save, generations_passed, correct_output_values

    generations_passed += 1
    time_taken = (time.time_ns() - start_time) / 1_000_000_000.0
    print(f"\nGeneration: {generations_passed} / {num_generations}")
    print("Time taken:", format_secs(time_taken))
    print("Estimated time left:", format_secs(time_taken /
          generations_passed * (num_generations - generations_passed)))

    # auto saves every 5 mins
    if time.time_ns() - last_save_time > 300e9:
        save_count += 1
        save_ga(ga_instance, ai_name)
        last_save_time = time.time_ns()

    correct_output_values = [random_wordle_word() for _ in range(10)]

    fitness_scores_since_save.append([])
    first_guesses_since_save.append([])


# function that needs to be called before using model.predict
def set_neural_network_weights(model, solution):
    # Fetch the parameters of the best solution.
    solution_weights = pygad.kerasga.model_weights_as_matrix(model=model,
                                                             weights_vector=solution)
    model.set_weights(solution_weights)


# r,y,g mapped to 0.0, 0.5, 1.0
def colour(word, solution):
    colours = [0, 0, 0, 0, 0]  # full gray default

    for i, (l1, l2) in enumerate(zip(word, solution)):
        if l1 == l2:
            colours[i] = 2  # green
        # special yellow rule: if number of l1 in solution is less than
        # the number of l1 in `word` up until l1 then make it gray instead of yellow
        elif l1 in solution and solution.count(l1) >= word[:i+1].count(l1):
            colours[i] = 1  # yellow

    return colours


# fitness function based on greens and yellows, currently unused
def coloured_fitness(predictions: list, data_outputs):
    colours = colour(predictions, data_outputs)
    return colours.count(2) + 0.5 * colours.count(1)


# returns the valid options from a list of words that match a (colours, letters) pair
def options_from_guess(possibilities: list, colours: list, guess: list):
    grays = [(l, i)
             for i, c, l in zip(range(5), colours, guess) if c == 0]

    yellows = defaultdict(lambda: 0)
    for i, c, l in zip(range(5), colours, guess):
        if c == 1:
            # makes sure the spot that's yellow can't be the guessed letter
            grays.append((l, i))
            yellows[l] += 1  # adds to the yellows-with-this-letter count

    greens = [(l, i)
              for i, c, l in zip(range(5), colours, guess) if c == 2]

    return [
        x for x in possibilities
        if all(x[i] != l for l, i in grays)
        and all(x.count(l) == c for l, c in yellows.items())
        and all(x[i] == l for l, i in greens)
    ]


def load_settings():
    with open(join("settings", "default"), "w") as file:
        file.write(Settings().to_json())

    settings = None
    available_settings = [f for f in os.listdir(
        "settings") if os.path.isfile(join("settings", f))]
    chosen_settings = "default"
    if len(available_settings) > 1:
        print("Choose one of the following settings files for training.")
        print("Available files: ", available_settings)
        chosen_settings = input(
            f"\nName of the file you wish to use: ")
        while chosen_settings not in available_settings:
            chosen_settings = input(
                "Invalid name. \nName of the file you wish to use: ")
    print(f"'{chosen_settings}'", "settings selected.")
    with open(join("settings", chosen_settings), "r") as file:
        settings = Settings.from_json(json.loads(file.read()))
    return settings


# saves the genetic algorithm and the training data
# returns the best fitness scores of every generation so far
def save_ga(ga_instance: pygad.GA, name: str):
    global fitness_scores_since_save, first_guesses_since_save, last_save_time

    print("\nSaving...")
    
    length_of_column = len(fitness_scores_since_save[1]) # required cause pygad is wack
    fitness_scores = copy.deepcopy([x[:length_of_column] for x in fitness_scores_since_save if len(x) != 0])
    first_guesses = copy.deepcopy([x[:length_of_column] for x in first_guesses_since_save if len(x) != 0])
    # initial value is time since last update, time before last update is added later
    time_trained = (time.time_ns() - last_save_time) / 1e9 % 300

    fitness_scores_since_save.clear()
    first_guesses_since_save.clear()

    os.makedirs(join("instances", name), exist_ok=True)
    if os.path.exists(join("instances", name, "training_data.txt")):
        with open(join("instances", name, "training_data.txt"), "r") as file:
            jl = json.loads(file.read())
            fitness_scores = jl["fitness_scores"] + fitness_scores
            first_guesses = jl["first_guesses"] + first_guesses
            time_trained += jl["time_trained"]
    with open(join("instances", name, "training_data.txt"), "w") as file:
        jd = json.dumps({"fitness_scores": fitness_scores,
                        "first_guesses": first_guesses,
                         "time_trained": time_trained})
        file.write(jd)

    ga_instance.save(join("instances", name, "algorithm"))
    
    return fitness_scores


# plots fitness values in a scatter plot with trendlines
# WARNING: fitness_scores not being a two-dimensional array will result in abstract errors
def plot_fitness_training(fitness_scores,
                 title="Generation vs. Fitness",
                 xlabel="Generation",
                 ylabel="Fitness",
                 linewidth=3,
                 font_size=14,
                 save_dir=None,
                 trendline=True):
    """
    Creates, shows, and returns a figure that summarizes how the fitness value evolved by generation. Can only be called after completing at least 1 generation. If no generation is completed, an exception is raised.

    Accepts the following:
        title: Figure title.
        xlabel: Label on the X-axis.
        ylabel: Label on the Y-axis.
        linewidth: Line width of the plot. Defaults to 3.
        font_size: Font size for the labels and title. Defaults to 14.
        plot_type: Type of the plot which can be either "plot" (default), "scatter", or "bar".
        color: Color of the plot which defaults to "#3870FF".
        trendline: Enable (True) or disable (False) the trendline.
        save_dir: Directory to save the figure.

    Returns the figure.
    """

    generations_completed = len(fitness_scores)

    if generations_completed < 1:
        raise RuntimeError(
            f"The plot_fitness() (i.e. plot_result()) method can only be called after completing at least 1 generation but ({generations_completed}) is completed.")

    indices = range(generations_completed)

    fig = plt.figure()
    
    plt.title(title, fontsize=font_size)
    plt.xlabel(xlabel, fontsize=font_size)
    plt.ylabel(ylabel, fontsize=font_size)
    
    best_fitness_scores = [np.max(fs) for fs in fitness_scores]
    
    for j, fs in enumerate(list(zip(*fitness_scores))):
        indices2, ffs = zip(*[(i, x) for i, x, b in zip(indices, fs, best_fitness_scores) if x != b])
        label = None
        if j == 0:
            label = "all scores"
        plt.scatter(indices2, ffs, color="r", edgecolor="none", alpha=0.3, label=label)
    
    plt.scatter(indices, best_fitness_scores, color="b", edgecolor="none", alpha=0.3, label="best scores")


    # trendline
    if trendline:
        z = np.polyfit(indices, [np.average(fs) for fs in fitness_scores], 1)
        p = np.poly1d(z)
        plt.plot(p(indices), linewidth=linewidth, color="r", label="average score trendline")
        
        z = np.polyfit(indices, best_fitness_scores, 1)
        p = np.poly1d(z)
        plt.plot(p(indices), linewidth=linewidth, color="b", label="best score trendline")

    if not save_dir is None:
        plt.savefig(fname=save_dir,
                                  bbox_inches='tight')
    plt.show()

    return fig
