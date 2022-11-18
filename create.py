from general import *
import general
from math import log2
from settings import Settings


model = tf.keras.models.Sequential()


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
        remaining_words = options_from_guess(
            remaining_words, colours, predictions)
        if len(remaining_words) == 0:
            return 0
        fitness += -log2(len(remaining_words) / total_words)

    best_fitness_scores[-1] = max(best_fitness_scores[-1], fitness)

    return fitness


def create(num_generations):
    global model
    
    general.num_generations = num_generations
    
    layers = int(input("Number of hidden layers (default = 100): "))
    nodes_per_layer = int(
        input("Number of nodes per hidden layer (default = 50): "))
    num_solutions = 8

    # Keras model
    model = tf.keras.models.Sequential([tf.keras.Input(shape=50)])
    for _ in range(layers):
        model.add(tf.keras.layers.Dense(nodes_per_layer, activation="sigmoid"))
    model.add(tf.keras.layers.Dense(5, activation="sigmoid"))
    
    keras_ga = pygad.kerasga.KerasGA(model=model, num_solutions=num_solutions)

    if (len(tf.config.list_physical_devices('GPU')) == 0):
        print("\nNo GPUs detected. Continuing on CPU.")

    # PyGAD parameters
    initial_population = keras_ga.population_weights

    os.makedirs("instances", exist_ok=True)
    os.makedirs("settings", exist_ok=True)
    with open(join("settings", "default"), "w") as file:
        file.write(Settings().to_json())

    settings = load_settings()

    ga_instance = settings.create_ga(num_generations=num_generations,
                                     initial_population=initial_population,
                                     fitness_func=fitness_func,
                                     on_generation=on_generation,
                                     on_stop=on_stop)

    name = input("AI name: ")
    while (os.path.exists(join("instances", name))):
        name = input(
            f"\nAI with the name `{name}` already exists.\nPlease enter a new name.\n\nAI name: ")
    
    model.save(join("instances", name, "model"))
    general.ai_name = name
    
    ga_instance.run()
    save_ga(ga_instance, name)
    plot_fitness(save_dir="fitness")


if (__name__ == "__main__"):
    generations = int(input("Number of generations to train for: "))
    if (generations >= 2):
        create(generations)
    else:
        print("The number of training generations needs to be greater than one.")
