from load import load
from create import create, fitness_func
from general import *
from matplotlib import pyplot as plt
import json
from math import log2
import os
from os.path import join
import pygad
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf


def train():
    print("Training selected.")

    training_type = None
    while training_type not in ["create", "load"]:
        training_type = input("Create a new AI or load one? [create/load]: ")

    generations = int(input("Number of generations to train for: "))
    if (generations >= 2):
        if (training_type == "create"):
            create(generations)
        elif (training_type == "load"):
            load(generations)
    else:
        print("The number of training generations needs to be greater than one.")


# creates a bar plot displaying the number of guesses in which the AI finished the game
def plot_finishes(finishes, save_dir):
    plt.title("Guesses per game", fontsize=14)
    plt.xlabel("Guess", fontsize=14)
    plt.ylabel("Count", fontsize=14)

    plt.bar(
        x=range(7), height=finishes)

    plt.savefig(fname=save_dir,
                              bbox_inches='tight')

    plt.show()


def plot_fitness(fitness_scores, save_dir=None):
    generations_completed = len(fitness_scores)

    if generations_completed < 1:
        raise RuntimeError(
            f"The plot_fitness() (i.e. plot_result()) method can only be called after completing at least 1 generation but ({generations_completed}) is completed.")
    
    plt.title("Generation vs. Fitness", fontsize=14)
    plt.xlabel("Generation", fontsize=14)
    plt.ylabel("Fitness", fontsize=14)

    indices = range(generations_completed)
    plt.scatter(indices, fitness_scores, color="r", edgecolor="none", alpha=0.3)

    if not save_dir is None:
        plt.savefig(fname=save_dir,
                                  bbox_inches='tight')
    plt.show()


# select the best solution/AI from a pool
# during training there are N AIs trained at a time,
# but for testing and playing only one is required so the best one is used
def select_best_solution(ga_instance: pygad.GA, model):
    # all solutions/AIs are given the same answers for fairness
    selected_words = [random_wordle_word() for _ in range(10)]
    
    # same principle as the fitness function in `create` and `load`
    def selection_fitness_func(solution, solution_idx):
        global possible_wordle_words, allowed_wordle_words
        total_fitness = 0.0
        
        set_neural_network_weights(model, solution)
        
        for j in range(10):
            input_values = [-1.0] * 50
            correct_output_values = selected_words[j]
            remaining_words = possible_wordle_words
            
            fitness = 0.0
            invalid_words = 0

            for i in range(6):
                # convert input values to tensor
                input_tensor = tf.convert_to_tensor(np.array([input_values]), dtype=tf.float32)
                
                # get AI output
                output_tensor = model(input_tensor)[0]
                output_values = list(np.floor(np.array(output_tensor) * 25.0))

                # mark the guess with wordle's grey, yellow, green colours
                # see paper for their meanings
                colours = colour(output_values, correct_output_values)

                # add current guess to the inputs for the next guess
                input_values[i*10:(i+1)*10] = output_values + colours

                prev_remaining_words_len = len(remaining_words)
                # calculate the possible remaining words
                remaining_words = options_from_guess(
                    remaining_words, colours, output_values)
                
                # if invalid word, reduce fitness
                if tuple(output_values) not in allowed_wordle_words:
                    invalid_words += 1
                
                # log2(0) returns undefined values, and the AI is supposed to avoid having no words left
                if len(remaining_words) == 0:
                    return 0.0 - invalid_words
                
                # information
                fitness += -log2(len(remaining_words) / prev_remaining_words_len)
                
                # if AI wins, break the loop
                if tuple(output_values) == correct_output_values:
                    fitness += 1.0
                    break
            
            fitness -= invalid_words
            total_fitness += fitness
        
        return total_fitness
    
    ga_instance.fitness_func = selection_fitness_func
    return ga_instance.best_solution()[0]


def test():
    print("Testing selected.")
    
    print("Testing will take several minutes and cannot be stopped once started.")
    choice = input("Continue? [Y/N]: ")
    while choice not in ["Y", "N"]:
        choice = input("Continue? [Y/N]: ")
    if choice == "N":
        pass
    
    print("Choose one of the following AIs for testing.")
    print("Available AIs:", os.listdir("instances"))
    name = input("AI instance to test: ")
    model: tf.keras.Model = tf.keras.models.load_model(
        join("instances", name, "model"))
    ga_instance: pygad.GA = pygad.load(join("instances", name, "algorithm"))
    
    solution = select_best_solution(ga_instance, model)
    set_neural_network_weights(model, solution)
    
    print("Testing started.")

    def modified_fitness_func(correct_output_values: tuple):
        global possible_wordle_words, allowed_wordle_words
        
        input_values = [-1.0] * 50
        remaining_words = possible_wordle_words
        
        fitness = 0.0
        invalid_words = 0

        for i in range(6):
            # convert input values to tensor
            input_tensor = tf.convert_to_tensor(np.array([input_values]), dtype=tf.float32)
            
            # get AI output
            output_tensor = model(input_tensor)[0]
            output_values = list(np.floor(np.array(output_tensor) * 25.0))

            # mark the guess with wordle's grey, yellow, green colours
            # see paper for their meanings
            colours = colour(output_values, correct_output_values)

            # add current guess to the inputs for the next guess
            input_values[i*10:(i+1)*10] = output_values + colours

            prev_remaining_words_len = len(remaining_words)
            # calculate the possible remaining words
            remaining_words = options_from_guess(
                remaining_words, colours, output_values)
            
            # if invalid word, reduce fitness
            if tuple(output_values) not in allowed_wordle_words:
                invalid_words += 1
            
            # log2(0) returns undefined values, and the AI is supposed to avoid having no words left
            if len(remaining_words) == 0:
                return (0.0 - invalid_words, 0)
            
            # information
            fitness += -log2(len(remaining_words) / prev_remaining_words_len)
            
            # if AI wins, break the loop
            if tuple(output_values) == correct_output_values:
                return (fitness - invalid_words + 1.0, i + 1)

        fitness -= invalid_words

        return (fitness, 0) # 0 means did not finish

    fitness_values, finishes = [], [0 for _ in range(7)]
    for i, w in enumerate(possible_wordle_words):
        if i % 10 == 0:
            print(f"\nGame: {i} / {len(possible_wordle_words)}")
            print(f"Percentage won: {1.0-finishes[0]/(i+1)*100.0:0.2f}")
        fv, rf = modified_fitness_func(w)
        fitness_values.append(fv)
        finishes[rf] += 1

    plot_fitness(fitness_values, save_dir=join(
        "instances", name, "testing_fitness"))
    plot_finishes(finishes, save_dir=join(
        "instances", name, "testing_finishes"))


def play():
    print("Playing selected.")
    
    print("Choose one of the following AIs for playing.")
    print("Available AIs:", os.listdir("instances"))
    name = input("AI instance to use: ")
    model: tf.keras.Model = tf.keras.models.load_model(
        join("instances", name, "model"))
    ga_instance: pygad.GA = pygad.load(join("instances", name, "algorithm"))
    
    print("Initializing... this may take a minute.")
    solution = select_best_solution(ga_instance, model)
    set_neural_network_weights(model, solution)
    print("Finished initializing.")

    print("How it works.")
    print("The Wordle AI will recommend guesses, which you need to use in your Wordle game.")
    print("After every guess, Wordle will colour the guessed word.")
    print("These colours may be any combination of grey, yellow and green.")
    print("You need to tell the AI these colours so it can give you a better guess.")
    print("\tGrey   = R")
    print("\tYellow = Y")
    print("\tGreen  = G")
    print("Examples: RRGYR, RRRRR, RGGGY, GGGGG, YYYRR")
    print("WARNING: The AI is *not* guaranteed to give you a valid guess.")
    
    while True:
        input_values = [-1.0] * 50
        
        for i in range(6):
            output_values = list(np.floor(model.predict(np.array([input_values]), verbose=0)[0] * 25.0))
            output_word = "".join(chr(int(l) + 97) for l in output_values)
            
            print(f"\nGuess {i + 1}.")
            print(f"The AI {name}'s recommended guess:", output_word)
            if i < 5:
                colour_strs = input(f"Colours Wordle gave the guess: ").lower()
                while any(c not in "ryg" for c in colour_strs) or len(colour_strs) != 5:
                    print("Invalid input. Please try again.")
                    colour_strs = input(f"Colours Wordle gave the guess: ").lower()
                colours = [ord(c) - 97 for c in colour_strs]
                input_values[i*10:(i+1)*10] = output_values + colours
        
        print("The game has ended.")
        
        cont = None
        while cont not in ["Y", "N"]:
            cont = input("Continue? [Y/N]: ").capitalize()
        
        if cont == "N":
            return


def plot():
    print("Plotting selected.")
    plt.title("Generation vs. Fitness", fontsize=14)
    plt.xlabel("Generation", fontsize=14)
    plt.ylabel("Fitness", fontsize=14)

    colour_values = "bgrcmykw"

    print("Available AIs:", os.listdir("instances"))
    count = int(input("Number of AIs to plot: "))
    while count > 4:
        print("Number of AIs must be less than 5.")
        count = int(input("Number of AIs to plot: "))
    for i in range(count):
        name = input(f"Name {i+1}: ")
        with open(join("instances", name, "training_data.txt"), "r") as file:
            jl = json.loads(file.read())
            fitness_scores: list = jl["fitness_scores"]
            indices = range(len(fitness_scores))

        for j, fs in enumerate(list(zip(*fitness_scores))):
            label, tl_label = None, None
            if j == 0:
                label, tl_label = f"{name} score", f"{name} score trendline"
            plt.scatter(indices, fs, color=colour_values[i], edgecolor="none", alpha=0.3, label=label)

            # trendline
            z = np.polyfit(indices, [np.average(fs) for fs in fitness_scores], 1)
            p = np.poly1d(z)
            plt.plot(p(indices), linewidth=3, color=colour_values[i], label=tl_label)

    plt.legend(title="Legend", title_fontproperties={"weight": "bold"})

    os.makedirs("figures", exist_ok=True)
    save = None
    while save not in ["Y", "N"]:
        save = input("Save figure? [Y/N]: ").capitalize()
    if save == "Y":
        save_name = input("Figure save name: ")
        plt.savefig(fname=join("figures", save_name),
                                  bbox_inches='tight')

    plt.show()


if __name__ == "__main__":
    print("Welcome.")
    print("Select a mode to continue.")
    print("For an explanation of all modes, type 'help'.")
    print("Available modes: [help, train, play, test, plot, quit]")
    mode = input("Mode: ")
    while mode not in ["help", "train", "play", "test", "plot", "quit"]:
        print("Invalid mode. Please try again.")
        mode = input("Mode: ")

    if mode == "help":
        print("\nModes:")
        print("help  - Show the modes' explanations as currently shown.")
        print("train - Train an AI. Options are 'create' and 'load' for training a new AI and training a previously created one respectively.")
        print("play  - Get AI guess recommendations for your Wordle game.")
        print("test  - Test how well an AI performs on all possible Wordle answers.")
        print("plot  - Plot the fitness scores one or more AIs in a graph.")
        print("")
        print("Terminology:")
        print("'guess'             - A five-letter word used in Wordle by the player.")
        print("'fitness score'     - A value used to rate how well an AI performs. The higher the better.")
        print("'fitness function'  - The function used to calculate the fitness score.")
        print("'generation'        - A stage in the development of an AI.")
        print("'genetic algorithm' - How the AIs evolve.")
        print("")
    if mode == "train":
        train()
    if mode == "plot":
        plot()
    if mode == "test":
        test()
    if mode == "play":
        play()
