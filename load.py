from general import *
from math import log2
import general
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf


model = tf.keras.models.Sequential()


def fitness_func(solution, solution_idx):
    global model
    global correct_output_words
    global possible_wordle_words
    global allowed_wordle_words
    global first_guesses_since_save
    global fitness_scores_since_save

    fitness = 0.0

    set_neural_network_weights(model, solution)

    for j in range(10): # TODO: check for the best possible value for this
        remaining_words = possible_wordle_words
        input_values = [0] * 725
        
        for i in range(6):
            # convert input values to tensor
            input_tensor = tf.convert_to_tensor(np.array([input_values]), dtype=tf.float32)
            
            # get AI output
            output_tensor = model(input_tensor)[0].numpy()
            output_index = np.argmax(output_tensor)
            output_word = allowed_wordle_words[output_index]
            
            # mark the guess with wordle's grey, yellow, green colours
            # see paper for their meanings
            colours = colour(output_word, correct_output_words[j])

            # add current guess to the inputs for the next guess
            input_values[i*145:(i+1)*145] = word_to_binary(output_word) + colours_to_binary(colours)

            prev_remaining_words_len = len(remaining_words)
            # calculate the possible remaining words
            remaining_words = options_from_guess(
                remaining_words, colours, output_word)

            # information
            fitness += -log2(len(remaining_words) / prev_remaining_words_len)

            # if AI wins, break the loop
            if output_word == correct_output_words[j]:
                fitness += 10.0
                break

            if i == 0 and j == 0: # store training data
                first_guesses_since_save[-1].add(output_word)

    # save training data
    fitness_scores_since_save[-1].append(fitness)

    return fitness / 10.0


def load(num_generations):
    global model
    
    print("Choose one of the following AIs for training.")
    print("Available AIs:", os.listdir("instances"))
    name = input("Name of the AI you wish to load: ")
    model = tf.keras.models.load_model(join("instances", name, "model"))
    
    if len(tf.config.list_physical_devices('GPU')) == 0:
        print("No GPUs detected. Continuing on CPU.")
    
    ga_instance: pygad.GA = pygad.load(join("instances", name, "algorithm"))
    
    # # buggy, training does not work properly after changing settings
    # change_settings = None
    # while change_settings not in ["Y", "N"]:
    #     change_settings = input("Change training settings? [Y/N]: ").capitalize()
    
    # if change_settings == "Y":
    #     settings = load_settings()
    #     settings.update_ga(ga_instance)
        
    #     with open(join("instances", name, "settings.txt"), "w") as file:
    #         file.write(settings.to_json())
    
    ga_instance.num_generations = num_generations
    ga_instance.fitness_func = fitness_func
    general.num_generations = num_generations
    general.ai_name = name
    general.ga = ga_instance
    os.system(f'title Command Prompt - py api.py - {name}') # set window title

    ga_instance.run()
    fitness_scores = save_ga(ga_instance, name)
    plot_fitness_training(fitness_scores, save_dir=join("instances", name, "fitness"))


if __name__ == "__main__":
    generations = int(input("Number of generations to train for: "))
    if generations >= 2:
        load(generations)
    else:
        print("The number of training generations needs to be greater than one.")
