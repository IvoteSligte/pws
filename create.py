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


def create(num_generations):
    general.num_generations = num_generations

    num_solutions = 8  # number of AI instances in a generation
    
    num_layers = input("Number of hidden layers (default = 3): ")
    if num_layers == "":
        num_layers = 3
    else:
        num_layers = int(num_layers)
    
    nodes_per_layer = input("Number of nodes per hidden layer (default = 725): ")
    if nodes_per_layer == "":
        nodes_per_layer = 725
    else:
        nodes_per_layer = int(nodes_per_layer)
    
    keep_parents = -1
    keep_elitism = 1

    # Keras model, stores layers/nodes
    layers = [tf.keras.Input(shape=725)]
    for _ in range(num_layers):
        layers.append(tf.keras.layers.Dense(nodes_per_layer, activation="relu"))
    layers.append(tf.keras.layers.Dense(len(allowed_wordle_words), activation="softmax"))
                  
    general.model = tf.keras.models.Sequential(layers)
    general.model.call = tf.function(general.model.call)

    keras_ga = pygad.kerasga.KerasGA(model=general.model, num_solutions=num_solutions)

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
        on_fitness=on_fitness,
        on_generation=on_generation,
        keep_parents=keep_parents,
        keep_elitism=keep_elitism,
    )
    general.ga = ga_instance
    # pygad provides a 10x slower function for random mutation, so it's overwritten
    # not recommended to change this, it will break things
    ga_instance.mutation = mutation_randomly_optimized

    name = input("AI name: ")
    while os.path.exists(join("instances", name)):
        name = input(
            f"\nAI with the name `{name}` already exists.\nPlease enter a new name.\n\nAI name: ")
    general.ai_name = name
    os.system(f'title Command Prompt - py api.py - {name}') # set window title

    general.model.save(join("instances", name, "model"))
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
    
    print("Training started.")
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
