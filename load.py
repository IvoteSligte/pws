from general import *
from math import log2
import general
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf


model = tf.keras.models.Sequential()


def fitness_func(solution, solution_idx):
    global model, correct_output_values, first_guesses_since_save, fitness_scores_since_save, possible_wordle_words, allowed_wordle_words

    input_values = [-1.0] * 50
    remaining_words = possible_wordle_words

    fitness = 0.0

    set_neural_network_weights(model, solution)

    for j in range(10):
        remaining_words = possible_wordle_words
        input_values = [-1.0] * 50
        
        for i in range(6):
            # get AI output
            output_values = list(np.floor(model.predict(
                np.array([input_values]), verbose=0)[0] * 25.0))

            # mark the guess with wordle's grey, yellow, green colours
            # see paper for their meanings
            colours = colour(output_values, correct_output_values[j])

            # add current guess to the inputs for the next guess
            input_values[i*10:(i+1)*10] = output_values + colours

            prev_remaining_words_len = len(remaining_words)
            # calculate the possible remaining words
            remaining_words = options_from_guess(
                remaining_words, colours, output_values)

            if i == 0:  # store training data
                first_guesses_since_save[-1].append(''.join(chr(int(l) + 97)
                                                            for l in output_values))

            # if invalid word, reduce fitness
            if tuple(output_values) not in allowed_wordle_words:
                fitness -= 1.0

            # log2(0) returns undefined values, and the AI is supposed to avoid having no words left
            if len(remaining_words) == 0:
                return 0.0  # TODO: remove invalid_words from the other fitness functions too if this works

            # information
            fitness += -log2(len(remaining_words) / prev_remaining_words_len)

            # if AI wins, break the loop
            if tuple(output_values) == correct_output_values[j]:
                fitness += 1.0
                break

    # save training data
    fitness_scores_since_save[-1].append(fitness)

    return fitness


def load(num_generations):
    global model
    
    print("Choose one of the following AIs for training.")
    print("Available AIs:", os.listdir("instances"))
    name = input("Name of the AI you wish to load: ")
    model = tf.keras.models.load_model(join("instances", name, "model"))
    
    if len(tf.config.list_physical_devices('GPU')) == 0:
        print("No GPUs detected. Continuing on CPU.")
    
    ga_instance: pygad.GA = pygad.load(join("instances", name, "algorithm"))
    
    change_settings = None
    while change_settings not in ["Y", "N"]:
        change_settings = input("Change training settings? [Y/N]: ").capitalize()
    
    if change_settings == "Y":
        settings = load_settings()
        settings.update_ga(ga_instance)
        
        with open(join("instances", name, "settings.txt"), "w") as file:
            file.write(settings.to_json())
    
    ga_instance.num_generations = num_generations
    ga_instance.fitness_func = fitness_func
    general.training_generations = num_generations
    general.ai_name = name
    
    ga_instance.run()
    fitness_scores = save_ga(ga_instance, name)
    plot_fitness_training(fitness_scores, save_dir=join("instances", name, "fitness"))


if __name__ == "__main__":
    generations = int(input("Number of generations to train for: "))
    if generations >= 2:
        load(generations)
    else:
        print("The number of training generations needs to be greater than one.")
