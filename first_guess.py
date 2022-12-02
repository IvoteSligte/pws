from math import log2
from general import *

start_time = time.time_ns()

information_scores = []

for i, a in enumerate(allowed_wordle_words):
    information_scores.append(sum(-log2(len(options_from_guess(possible_wordle_words, colour(a, p), a)) / len(possible_wordle_words))
                                  for p in possible_wordle_words) / len(possible_wordle_words))
    time_taken = (time.time_ns() - start_time) / 1e9
    time_left = (2309 - (i + 1)) * (time_taken / (i + 1))
    print(f"{i} / 2309 | Time taken: {format_secs(time_taken)} | Estimated time left: {format_secs(time_left)}")

information_scores = [sum(-log2(len(options_from_guess(possible_wordle_words, colour(a, p), a)) / len(possible_wordle_words))
                          for p in possible_wordle_words) / len(possible_wordle_words) for a in allowed_wordle_words]

sorted_indices = np.flip(np.argsort(information_scores))

sorted_information_scores = np.array(information_scores)[sorted_indices]
sorted_allowed_words = np.array(allowed_wordle_words)[sorted_indices]

with open("sorted_first_guesses.txt", "w") as file:
    file.write(filter(lambda x: x not in "()", "\n".join(
        zip(sorted_allowed_words, sorted_information_scores))))
