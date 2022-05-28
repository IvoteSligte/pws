import csv


# param: codes = tuple/list - list of tuples with letters and codes e.g. ("w", "g")
def analyse(codes):
    with open("word_list.csv", newline='') as csvfile:
        reader = csv.reader(csvfile)
        remaining = [csvword for csvword in reader if matches(codes, csvword)]
        print(remaining)


def matches(word, codes):
    # green  - g = in word and on the spot
    # yellow - y = in word but not on the spot
    # grey   - r = not in the word or yellow/green in another spot
    # none   - n = no data about spot

    # grey with greens means nowhere besides where there are greens
    # grey with yellows means just not in that spot

    for i, (l, c) in enumerate(codes):
        if c == 'g':  # green
            if word[i] != l:
                return False
        elif c == 'y':  # yellow
            if l not in word or word[i] == l:
                return False
        elif c == 'r':  # grey
            if word[i] == l:
                return False
            if (
                (l, 'y') not in codes and
                any(x == l for j, x in enumerate(word) if codes[j][1] not in ('n', 'g'))
            ):
                return False
    return True
