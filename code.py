from collections import defaultdict
from itertools import product
from math import log2
from pprint import pprint


def entropy(target, possible_words):
    iterable = (tuple(set(analyse(possible_words, target, 'n' * i + c)) for c in 'gyr') for i in range(len(target)))
    total = len(possible_words)
    amount = 0
    for prod in product(*iterable):
        count = len(set.intersection(*prod))
        if count > 0:
            amount += count / total * information(count, total)
    return amount


def information(possible_word_count, total_word_count):
    if possible_word_count == 0:
        return 'math error'
    return -log2(possible_word_count / total_word_count)


def multi_analyse(possible_words, *pairs):
    for target, hints in pairs:
        possible_words = analyse(possible_words, target, hints)
    return possible_words


def analyse(possible_words, target, hints):
    return list(filter(lambda w: is_match(w, target, hints), possible_words))


def is_match(word, target, hints):
    # green  - g = in word and on the spot
    # yellow - y = in word but not on the spot
    # grey   - r = not in the word or yellow/green in another spot
    # none   - n = no data about spot

    # grey with greens means nowhere besides where there are greens
    # grey with yellows means just not in that spot

    includes = defaultdict(lambda: 0)
    for x, h in zip(target, hints):
        if h in 'gy':
            includes[x] += 1
        elif h == 'r':
            includes[x] -= 1

    has_counts = all(word.count(k) == max(v, 0) for k, v in includes.items())

    char_matches = all(x == y if h == 'g' else x != y for x, y, h in zip(target, word, hints) if h in 'gyr')

    return has_counts and char_matches


if __name__ == '__main__':
    with open("possible_words.txt") as file:
        words = [word.rstrip() for word in file]
        # print(entropy(input("Input word: "), words))

        inputs = [
            (input("\nInput word: "), input("Input code: "))
            for _ in range(int(input("Number of inputs: ")))
        ]
        results = multi_analyse(words, *inputs)
        print("\nPossible words:")
        pprint(results)
        print("\nInformation:", information(len(results), len(words)))
