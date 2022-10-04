import input_set
import numpy as np

ORGANISM_COUNT = 16
INPUT_COUNT = 1000  # set length

INPUT_SIZE = 50  # number of input nodes
OUTPUT_SIZE = 5  # number of output nodes

N = 60  # nodes per layer
L = 10  # layers

NODE_COUNT = INPUT_SIZE + L * N + OUTPUT_SIZE

def init_organism():
    return [[], input_set.random_word()]

organism_data = [init_organism() for _ in range(ORGANISM_COUNT)]


def sigmoid(w):
    p = np.exp(w)
    p = p / (1 + p)
    return p


def calculate_layer(data, solution, A, B, activation):
    return [activation(np.sum(np.array(data) * np.array(solution[i*A:(i+1)*A])) + solution[A*B+i]) for i in range(B)]


def fitness_func(solution, solution_idx):
    [inputs, answers] = organism_data[solution_idx]
    
    data = inputs + [0 for _ in range(INPUT_SIZE - len(inputs))]
    
    data = calculate_layer(data, solution, INPUT_SIZE, N, lambda x: max(x, 0))
    solution = solution[(INPUT_SIZE+1)*N+1:]

    for i in range(1, L):
        data = calculate_layer(data, solution, N, N, lambda x: max(x, 0))
        solution = solution[(N+1)*N+1:]

    data = calculate_layer(data, solution, N, OUTPUT_SIZE, lambda x: np.floor(sigmoid(x) * 26.0))
    
    colours = input_set.colour(data, answers)
    
    if (len(organism_data[solution_idx][0]) < 50):
        organism_data[solution_idx][0].extend(data + list(map(lambda x: x / 2, colours)))
    else:
        organism_data[solution_idx] = init_organism()

    return colours.count(2)
