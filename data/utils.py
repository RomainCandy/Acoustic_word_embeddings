import numpy as np
import random
import torch


def random_padding(array: np.array, width_final: int) -> np.array:
    """

    Args:
        array: mfcc shape (length, nb_acoustic)
        width_final: final length

    Returns:
        np.array shape(width_final, nb_acoustic) with random 0 padding
    """

    current_width = array.shape[0]
    where_to_pad = np.random.choice(range(width_final - current_width))
    res = np.zeros((width_final, array.shape[1]))
    res[where_to_pad: (where_to_pad + current_width)] = array
    return res


def choose_words(all_words, word):
    return random.choice(all_words[word])


def transform(array):
    return torch.from_numpy(array).float()


def get_all_word(file: str)-> dict:
    """

    Args:
        file: word_i:word_{i1},word_{i2}...

    Returns:
        {word_i:[word_{i1}, word_{i2}...]}

    """
    res = {}
    with open(file, 'r') as f:
        for line in f:
            line = line.rstrip(',\n')
            word, l_word = line.split(':')
            res[word] = l_word.split(',')
    return res
