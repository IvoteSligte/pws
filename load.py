from general import *
import general
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf


def fitness_func(solution, solution_idx):
    global possible_wordle_words
    global allowed_wordle_words
    global first_guesses_since_save

    set_neural_network_weights(general.model, solution)
    
    # calculate first guess and add it to the training data
    input_tensor = tf.convert_to_tensor(np.array([[0] * 725]), dtype=tf.float32)
    output_word = allowed_wordle_words[np.argmax(general.model(input_tensor)[0].numpy())]
    first_guesses_since_save[-1].add("".join(chr(l) for l in output_word))
    
    return sum(map(fitness_func_core, possible_wordle_words)) + max_fitness_per_word * possible_wordle_words.size


def load(num_generations):    
    print("Choose one of the following AIs for training.")
    print("Available AIs:", os.listdir("instances"))
    name = input("Name of the AI you wish to load: ")
    general.model = tf.keras.models.load_model(join("instances", name, "model"))
    
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

    print("Training started.")
    ga_instance.run()
    fitness_scores = save_ga(ga_instance, name)
    plot_fitness_training(fitness_scores, save_dir=join("instances", name, "fitness"))


if __name__ == "__main__":
    generations = int(input("Number of generations to train for: "))
    if generations >= 2:
        load(generations)
    else:
        print("The number of training generations needs to be greater than one.")
