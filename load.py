from general import *
import general
from math import log2


def fitness_func(solution, solution_idx):
    global model

    data_inputs = []
    data_outputs = random_word()

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

    best_fitness_scores[-1] = max(best_fitness_scores[-1], fitness)

    return fitness

name = input("Name of the AI you wish to load: ")
model: tf.keras.Model = tf.keras.models.load_model(join("instances", name, "model"))

def load(num_generations):
    if (len(tf.config.list_physical_devices('GPU')) == 0):
        print("No GPUs detected. Continuing on CPU.")
    
    ga_instance: pygad.GA = pygad.load(join("instances", name, "algorithm"))
    
    change_settings = None
    while (change_settings not in ["Y", "N"]):
        change_settings = input("Change settings? [Y/N]: ").capitalize()
    
    if (change_settings == "Y"):
        settings = load_settings()
        settings.update_ga(ga_instance)
    
    ga_instance.num_generations = num_generations
    general.num_generations = num_generations
    general.ai_name = name
    
    ga_instance.run()
    save_ga(ga_instance, name)
    plot_fitness(save_dir="fitness")


if (__name__ == "__main__"):
    generations = int(input("Number of generations to train for: "))
    if (generations >= 2):
        load(generations)
    else:
        print("The number of training generations needs to be greater than one.")
