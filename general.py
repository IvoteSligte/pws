from collections import defaultdict
import time
import pygad
import pygad.kerasga
import os
from os.path import join
from settings import Settings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
import numpy as np
import random
import json
import matplotlib


wordle_words = [[ord(l) - 97 for l in word.rstrip()]
                for word in open("possible_words.txt")]


random_generator = random.Random(time.time() * 1000)


num_generations = None
best_fitness_scores = [0]
start_time = time.time_ns()
save_counter = 1
ai_name = None


def on_generation(ga_instance):
    global save_counter
    
    time_taken = (time.time_ns() - start_time) / 1e9
    generations_passed = len(best_fitness_scores)
    print("\nGeneration:", generations_passed)
    print("Time taken:", round(time_taken, 1), "s")
    print("Estimated time left:", round(time_taken / generations_passed * (num_generations - generations_passed), 1), "s")
    
    # auto saves every 10 mins
    if (time_taken / 600 > save_counter):
        save_counter += 1
        save_ga(ga_instance, ai_name)

    best_fitness_scores.append(0)


def on_stop(ga_instance, fitnesses_of_last_generation):
    best_fitness_scores.pop()


def random_word():
    global random_generator
    return wordle_words[random_generator.randrange(len(wordle_words))]


# pygad function
def predict(model, solution, data):
    # Fetch the parameters of the best solution.
    solution_weights = pygad.kerasga.model_weights_as_matrix(model=model,
                                               weights_vector=solution)
    model.set_weights(solution_weights)
    predictions = model.predict(data, verbose=0)

    return predictions


# r,y,g to 0.0, 0.5, 1.0
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


def coloured_fitness(predictions: list, data_outputs):
    colours = colour(predictions, data_outputs)
    return colours.count(2) + 0.5 * colours.count(1)


# returns the valid options from a list of words that match a (colours, letters) pair
def options_from_guess(possibilities: list, colours: list, guess: list):
    grays = [(l, i)
             for (i, c, l) in zip(range(5), colours, guess) if c == 0]

    yellows = defaultdict(lambda: 0)
    for (i, c, l) in zip(range(5), colours, guess):
        if c == 1:
            # makes sure the spot that's yellow can't be the guessed letter
            grays.append((l, i))
            yellows[l] += 1  # adds to the yellows-with-this-letter count

    greens = [(l, i)
              for (i, c, l) in zip(range(5), colours, guess) if c == 2]

    return [
        x for x in possibilities
        if all(x[i] != l for (l, i) in grays)
        and all(x.count(l) == c for (l, c) in yellows.items())
        and all(x[i] == l for (l, i) in greens)
    ]


def load_settings():
    settings = None
    available_settings = [f for f in os.listdir(
        "settings") if os.path.isfile(join("settings", f))]
    chosen_settings = input(
        f"\nChoose one of the following settings files for training.\nAvailable files: {available_settings}\n\nName of the file you wish to use: ")
    with open(join("settings", chosen_settings), "r") as file:
        settings = Settings.from_json(json.loads(file.read()))
    return settings


def save_ga(ga_instance: pygad.GA, name: str):
    global best_fitness_scores
    
    print("\nSaving...")
    print("Average fitness of every generation: " +
          str(np.average(best_fitness_scores)))

    os.makedirs(join("instances", name), exist_ok=True)
    if (os.path.exists(join("instances", name, "best_fitness_scores.txt"))):
        with open(join("instances", name, "best_fitness_scores.txt"), "r") as file:
            jl = json.loads(file.read())
            best_fitness_scores = jl["best_fitness_scores"] + best_fitness_scores
    with open(join("instances", name, "best_fitness_scores.txt"), "w") as file:
        jd = json.dumps({"best_fitness_scores": best_fitness_scores})
        file.write(jd)

    ga_instance.save(join("instances", name, "algorithm"))


def plot_fitness(title="PyGAD - Generation vs. Fitness", 
                    xlabel="Generation", 
                    ylabel="Fitness", 
                    linewidth=3, 
                    font_size=14, 
                    plot_type="plot",
                    color="#3870FF",
                    save_dir=None):

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
            save_dir: Directory to save the figure.

        Returns the figure.
        """

        generations_completed = len(best_fitness_scores)

        if generations_completed < 1:
            raise RuntimeError(f"The plot_fitness() (i.e. plot_result()) method can only be called after completing at least 1 generation but ({generations_completed}) is completed.")

#        if self.run_completed == False:
#            if not self.suppress_warnings: warnings.warn("Warning calling the plot_result() method: \nGA is not executed yet and there are no results to display. Please call the run() method before calling the plot_result() method.\n")

        indices = range(generations_completed)

        fig = matplotlib.pyplot.figure()
        if plot_type == "plot":
            matplotlib.pyplot.plot(best_fitness_scores, linewidth=linewidth, color=color)
        elif plot_type == "scatter":
            matplotlib.pyplot.scatter(indices, best_fitness_scores, linewidth=linewidth, color=color)
        elif plot_type == "bar":
            matplotlib.pyplot.bar(indices, best_fitness_scores, linewidth=linewidth, color=color)
        matplotlib.pyplot.title(title, fontsize=font_size)
        matplotlib.pyplot.xlabel(xlabel, fontsize=font_size)
        matplotlib.pyplot.ylabel(ylabel, fontsize=font_size)
        
        # trendline
        z = np.polyfit(indices, best_fitness_scores, 1)
        p = np.poly1d(z)
        matplotlib.pyplot.plot(p(indices), linewidth=linewidth, color="#D2042D", alpha=0.5)
        
        if not save_dir is None:
            matplotlib.pyplot.savefig(fname=save_dir, 
                                      bbox_inches='tight')
        matplotlib.pyplot.show()

        return fig


# known bugs: training sometimes freezes when an instance with the same settings already exists
