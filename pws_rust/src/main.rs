use rayon::prelude::*;
use std::{collections::HashMap, fs};

fn argsort(data: &[f32]) -> Vec<usize> {
    let mut indices = (0..data.len()).collect::<Vec<_>>();
    indices.sort_by(|&i, &j| data[i].total_cmp(&data[j]));
    indices
}

fn colour(word: &[u8], solution: &[u8]) -> [u8; 5] {
    let mut colours = [0u8; 5];

    for (i, (l, s)) in word.iter().zip(solution.iter()).enumerate() {
        if l == s {
            colours[i] = 2;
        } else if solution.contains(l)
            && solution.iter().filter(|c| *c == l).count()
                >= word[0..=i].iter().filter(|c| *c == l).count()
        {
            colours[i] = 1;
        }
    }

    colours
}

fn options_from_guess<'a>(
    possibilities: &'a Vec<Vec<u8>>,
    colours: &[u8],
    guess: &[u8],
) -> Vec<&'a Vec<u8>> {
    let mut grays = vec![];
    let mut greens = vec![];

    let mut yellows = HashMap::new();
    for (i, (c, l)) in colours.iter().zip(guess.iter()).enumerate() {
        match c {
            0 => grays.push((i, l)),
            1 => {
                grays.push((i, l));
                *yellows.entry(l).or_insert(0) += 1;
            }
            2 => greens.push((i, l)),
            _ => (),
        }
    }

    possibilities
        .iter()
        .filter(|w| {
            let r = grays.iter().all(|(i, l)| w[*i] != **l);
            let g = greens.iter().all(|(i, l)| w[*i] == **l);
            let y = yellows
                .iter()
                .all(|(l, c)| w.iter().filter(|x| x == l).count() >= *c);
            r && g && y
        })
        .collect::<Vec<_>>()
}

fn read_wordle_words(path: &str) -> Vec<Vec<u8>> {
    fs::read(path)
        .unwrap()
        .into_iter()
        .filter(|b| '\n' as u8 != *b && '\r' as u8 != *b)
        .collect::<Vec<_>>()
        .chunks_exact(5) // assumes the file has *only* five-letter words
        .map(|chunk| chunk.to_vec())
        .collect::<Vec<Vec<u8>>>()
}

fn main() {
    let allowed_words_bytes = read_wordle_words("..//allowed_words.txt");
    let possible_words_bytes = read_wordle_words("..//possible_words.txt");

    println!("Starting calculations. No further messages will be provided until all are finished.");

    let information_scores = allowed_words_bytes
        .par_iter()
        .map(|a| {
            let total_information: f32 = possible_words_bytes
                .iter()
                .map(|p| {
                    let colours = colour(a, p);
                    let remaining_words =
                        options_from_guess(&possible_words_bytes, &colours, a).len();
                    -(remaining_words as f32 / possible_words_bytes.len() as f32).log2()
                })
                .sum();

            total_information / possible_words_bytes.len() as f32
        })
        .collect::<Vec<_>>();

    println!("Finished all. Saving...");

    let mut sorted_indices = argsort(&information_scores);
    sorted_indices.reverse();

    let sorted_data = sorted_indices
        .par_iter()
        .map(|i| {
            let word =
                String::from_utf8(allowed_words_bytes[*i].clone()).expect("invalid utf-8 sequence");
            format!("{}, {}", word, information_scores[*i])
        })
        .collect::<Vec<String>>();

    fs::write("..//sorted_first_guesses.txt", sorted_data.join("\n")).unwrap();
}
