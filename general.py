from collections import defaultdict
import copy
from math import floor, log2
import time
import pygad
import pygad.kerasga
import os
from os.path import join
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
from settings import Settings
import numpy as np
import random
import json
from matplotlib import pyplot as plt


possible_wordle_words = np.array([np.array([ord(l) for l in word.rstrip()]) for word in open("possible_words.txt")], dtype=np.int8)
allowed_wordle_words = np.array([np.array([ord(l) for l in word.rstrip()]) for word in open("allowed_words.txt")], dtype=np.int8)


def random_wordle_word():
    global possible_wordle_words
    return np.random.choice(possible_wordle_words, size=1)


model: tf.keras.models.Sequential = None
num_generations = None
generations_passed = 0
fitness_scores_since_save = []
first_guesses_since_save = [set()]
start_time = 0
last_save_time = 0
save_count = 0
ai_name = None

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


def on_fitness(ga_instance, fitness_scores):
    global fitness_scores_since_save
    fitness_scores_since_save.append(list(fitness_scores))


def on_generation(ga_instance):
    global save_count
    global last_save_time
    global first_guesses_since_save
    global generations_passed

    generations_passed += 1
    time_taken = (time.time_ns() - start_time) / 1e9
    print(f"\nGeneration: {generations_passed} / {num_generations}")
    print("Time taken:", format_secs(time_taken))
    print("Estimated time left:", format_secs(time_taken /
          generations_passed * (num_generations - generations_passed)))

    # auto saves every 5 mins
    if time.time_ns() - last_save_time > 300e9:
        save_count += 1
        save_ga(ga_instance, ai_name)
        last_save_time = time.time_ns()

    first_guesses_since_save.append(set())


def fitness_func_core(correct_output_word):
    global model
    global possible_wordle_words
    global allowed_wordle_words
    
    remaining_words = possible_wordle_words.copy()
    input_values = np.array([])
    
    for _ in range(6):
        # convert input values to tensor
        input_tensor = tf.convert_to_tensor([np.pad(input_values, (0, 725-len(input_values)), 'constant')])
        
        # get AI output
        output_word = allowed_wordle_words[np.argmax(model(input_tensor)[0].numpy())]
        
        # if AI wins, break the loop
        if np.array_equal(output_word, correct_output_word):
            return 10.0
        
        # mark the guess with wordle's grey, yellow, green colours
        # see paper for their meanings
        colours = colour(output_word, correct_output_word)
        
        # add current guess to the inputs for the next guess
        input_values = np.concatenate((input_values, word_to_binary(output_word), colours_to_binary(colours)))
        
        # calculate the possible remaining words
        remaining_words = options_from_guess(
            remaining_words, colours, output_word)
    
    return -log2(len(remaining_words) / len(possible_wordle_words))


# function that needs to be called before using model.predict
def set_neural_network_weights(model, solution):
    # Fetch the parameters of the best solution.
    solution_weights = pygad.kerasga.model_weights_as_matrix(model=model,
                                                             weights_vector=solution)
    model.set_weights(solution_weights)


# returns a list of boolean integers representing the binary form of a word
def word_to_binary(word):
    return np.array([np.concatenate((np.zeros(l - 97), np.ones(1), np.zeros(25 - (l - 97)))) for l in word]).flatten()


# returns a list of boolean integers representing the binary form of a set of RYG colours
def colours_to_binary(colours):
    return np.array([np.concatenate((np.zeros(c), np.ones(1), np.zeros(2 - c))) for c in colours]).flatten()


# r,y,g mapped to 0, 1, 2
def colour(word: np.ndarray, solution: np.ndarray):
    colours = np.array([0, 0, 0, 0, 0]) # full gray default
    
    greens = word == solution
    colours[greens] = 2 # green
    
    # number of l in solution which are not green = total possible yellows of letter l
    # and must be > the number of yellows so far of letter l
    # tldr; number of yellows left must be > 0
    yellows = np.array([not greens[i] and np.count_nonzero((solution == l) & np.logical_not(greens)) > np.count_nonzero(((word == l) & np.logical_not(greens))[:i]) for i, l in enumerate(word)])
    colours[yellows] = 1

    return colours


# returns the valid options from a list of words that match a (colours, letters) pair
def options_from_guess(words: np.ndarray, colours: np.ndarray, guess: np.ndarray):
    prev_words = words
    for i, c, l in zip(range(5), colours, guess):
        if c == 0:
            # grey letter cannot in be in the word, unless there are greens/yellows of that letter (but still not at this index)
            words = words[words[:, i] != l]
            words = words[np.count_nonzero(words == l, axis=-1) == np.count_nonzero(colours[guess == l] >= 1)]
        elif c == 1:
            # makes sure the spot that's yellow can't be the guessed letter
            words = words[words[:, i] != l]
            # number of yellows of a certain letter <= the amount of the letter in the solution
            words = words[np.count_nonzero(words == l, axis=-1) >= np.count_nonzero(colours[guess == l] >= 1)]
        elif c == 2:
            # green letter has to be in the word
            words = words[words[:, i] == l]
    
    return words


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
    print(f"'{chosen_settings}'", "settings selected.")
    with open(join("settings", chosen_settings), "r") as file:
        settings = Settings.from_json(json.loads(file.read()))
    return settings


# saves the genetic algorithm and the training data
# returns the best fitness scores of every generation so far
def save_ga(ga_instance: pygad.GA, name: str):
    global fitness_scores_since_save, first_guesses_since_save, last_save_time

    print("\nSaving...")
    
    fitness_scores = copy.deepcopy(fitness_scores_since_save)
    first_guesses = copy.deepcopy([list(x) for x in first_guesses_since_save]) # sets cannot be JSON serialized
    if len(first_guesses[-1]) == 0:
        first_guesses.pop()
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


# modified pygad function
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

    fig = plt.figure()
    
    plt.title(title, fontsize=font_size)
    plt.xlabel(xlabel, fontsize=font_size)
    plt.ylabel(ylabel, fontsize=font_size)
    
    best_fitness_scores_raw = [np.max(fs) if len(fs) > 0 else -1e20 for fs in fitness_scores]
    
    all_indices, all_fitness_scores = zip(*[(i, x) for i, s in enumerate(fitness_scores) for x in s])
    
    other_indices, other_fitness_scores = zip(*[(i, x) for i, x in zip(all_indices, all_fitness_scores) if best_fitness_scores_raw[i] != x])
    plt.scatter(other_indices, other_fitness_scores, color="r", edgecolor="none", alpha=0.3, label="all scores")
    
    best_indices, best_fitness_scores_raw = zip(*[(i, x) for i, x in enumerate(best_fitness_scores_raw) if x != -1e20]);
    plt.scatter(best_indices, best_fitness_scores_raw, color="b", edgecolor="none", alpha=0.3, label="best scores")

    # trendline
    if trendline:
        z = np.polyfit(all_indices, all_fitness_scores, 1)
        p = np.poly1d(z)
        plt.plot(p(range(len(fitness_scores))), linewidth=linewidth, color="r", label="all scores trendline")
        
        z = np.polyfit(best_indices, best_fitness_scores_raw, 1)
        p = np.poly1d(z)
        plt.plot(p(range(len(fitness_scores))), linewidth=linewidth, color="b", label="best scores trendline")

    plt.legend(title="Legend", title_fontproperties={"weight": "bold"})

    if not save_dir is None:
        plt.savefig(fname=save_dir,
                                  bbox_inches='tight')
    plt.show()

    return fig


ga: pygad.GA = None
# modified pygad function
def mutation_randomly_optimized(offspring):
        global ga
        
        """
        Applies the random mutation the mutation probability. For each gene, if its probability is <= that mutation probability, then it will be mutated randomly.
        It accepts a single parameter:
            -offspring: The offspring to mutate.
        It returns an array of the mutated offspring.
        """

        # Random mutation changes one or more genes in each offspring randomly.
        for offspring_idx in range(offspring.shape[0]):
            mutation_indices = np.array(random.sample(range(0, ga.num_genes), ga.mutation_num_genes))
            # Generating random values.
            random_values = np.random.uniform(low=ga.random_mutation_min_val, 
                                                    high=ga.random_mutation_max_val, 
                                                    size=len(mutation_indices))
            
            # If the mutation_by_replacement attribute is False, then the random value is added to the gene value.
            if not ga.mutation_by_replacement:
                random_values += offspring[offspring_idx, mutation_indices]
            
            offspring[offspring_idx, mutation_indices] = random_values

        return offspring
