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

    for j in range(10):
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


def create(num_generations):
    global model

    general.num_generations = num_generations

    num_solutions = 8  # number of AI instances in a generation
    
    num_layers = input("Number of hidden layers (default = 10): ")
    if num_layers == "":
        num_layers = 10
    else:
        num_layers = int(num_layers)
    
    nodes_per_layer = input("Number of nodes per hidden layer (default = 725): ")
    if nodes_per_layer == "":
        nodes_per_layer = 725
    else:
        nodes_per_layer = int(num_layers)
    
    keep_parents = -1
    keep_elitism = 1

    # Keras model, stores layers/nodes
    layers = [tf.keras.Input(shape=725)]
    for _ in range(num_layers):
        layers.append(tf.keras.layers.Dense(nodes_per_layer, activation="sigmoid")) # TODO:try ReLU
    layers.append(tf.keras.layers.Dense(len(allowed_wordle_words), activation="softmax"))
    model = tf.keras.models.Sequential(layers)
    model.call = tf.function(model.call)

    keras_ga = pygad.kerasga.KerasGA(model=model, num_solutions=num_solutions)

    if len(tf.config.list_physical_devices('GPU')) == 0:
        print("\nNo GPUs detected. Continuing on CPU.")

    # PyGAD parameters
    initial_population = keras_ga.population_weights

    os.makedirs("instances", exist_ok=True)
    os.makedirs("settings", exist_ok=True)

    settings = load_settings()

    ga_instance = settings.create_ga(
        num_generations=num_generations,
        initial_population=initial_population,
        fitness_func=fitness_func,
        on_start=on_start,
        on_generation=on_generation,
        keep_parents=keep_parents,
        keep_elitism=keep_elitism,
    )

    name = input("AI name: ")
    while os.path.exists(join("instances", name)):
        name = input(
            f"\nAI with the name `{name}` already exists.\nPlease enter a new name.\n\nAI name: ")
    general.ai_name = name
    os.system(f'title Command Prompt - py api.py - {name}') # set window title

    model.save(join("instances", name, "model"))
    with open(join("instances", name, "creation_data.txt"), "w") as file:
        jd = json.dumps({
            "layers": num_layers,
            "nodes_per_layer": nodes_per_layer,
            "num_solutions": num_solutions, 
            "keep_parents": keep_parents,
            "keep_elitism": keep_elitism,
        })
        file.write(jd)
    
    with open(join("instances", name, "settings.txt"), "w") as file:
        file.write(settings.to_json())

    ga_instance.run()
    fitness_scores = save_ga(ga_instance, name)
    plot_fitness_training(fitness_scores, save_dir=join(
        "instances", name, "fitness"))


if __name__ == "__main__":
    generations = int(input("Number of generations to train for: "))
    if generations >= 2:
        create(generations)
    else:
        print("The number of training generations needs to be greater than one.")
