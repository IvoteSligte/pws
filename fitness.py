import input_set
import numpy as np

ORGANISM_COUNT = 16
INPUT_COUNT = 1000  # set length

INPUT_SIZE = 50  # number of input nodes
OUTPUT_SIZE = 5  # number of output nodes

N = 20  # nodes per layer
L = 10  # layers

NODE_COUNT = INPUT_SIZE + L * N + OUTPUT_SIZE


def sigmoid(w):
    p = np.exp(w)
    p = p / (1 + p)
    return p


def calculate_layer(data, solution, A, B, activation):
    return [activation(np.sum(np.array(data) * np.array(solution[i*A:(i+1)*A])) + solution[A*B+i]) for i in range(B)]


def fitness_func(solution, solution_idx):
    [inputs, answers] = input_set.gen_input()
    
    data = calculate_layer(inputs, solution, INPUT_SIZE, N, lambda x: max(x, 0))
    solution = solution[(INPUT_SIZE+1)*N+1:]

    for i in range(1, L):
        data = calculate_layer(data, solution, N, N, lambda x: max(x, 0))
        solution = solution[(N+1)*N+1:]

    data = calculate_layer(data, solution, N, OUTPUT_SIZE, lambda x: np.floor(sigmoid(x) * 26.0))
    
    colours = input_set.colour(data, answers)

    return colours.count(2)
