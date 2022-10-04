import random
import numpy as np

WORDS = [[ord(l) - 97 for l in word.rstrip()]
         for word in open("possible_words.txt")]


def random_word():
    return WORDS[random.randrange(len(WORDS))]


def gen_input():
    solution = random_word()

    guesses = []
    data = []
    for _ in range(random.randrange(5)):
        word = random_word()
        if word not in guesses:
            colours = colour(word, solution)
            data.extend(list(np.array(word) / 25) + list(np.array(colours) / 2))
            guesses.append(word)

    data.extend([0 for _ in range(50 - len(data))])

    return (data, solution)


# r,y,g to 0.0, 0.5, 1.0
def colour(word, solution):
    colours = [0, 0, 0, 0, 0]  # full gray default

    for i, (l1, l2) in enumerate(zip(word, solution)):
        if l1 == l2:
            colours[i] = 2  # green
        # special yellow rule: if number of l1 in solution is less than
        # the number of l1 in `word` up until l1 then make it gray instead of yellow
        elif l1 in solution and solution.count(l1) >= word[:i+1].count(l1):
            colours[i] = 1  # yellow

    return colours


if __name__ == '__main__':
    print(gen_input())
