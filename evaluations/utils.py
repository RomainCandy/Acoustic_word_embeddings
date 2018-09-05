from collections import Counter
from itertools import combinations

import os
import pandas as pd
import torch.nn.functional as F


def create_directory(path: str)-> None:
    """

    Args:
        path: str representing a directory path

    Returns:
        if this directory doesn't exist then create it
    """
    if not os.path.exists(path):
        os.makedirs(path)


def get_label_to_word(lab_to_word):
    """
    Returns:
        {idword:word}
    """
    dataframe = pd.read_csv(lab_to_word, header=None)
    dataframe.columns = ['labels', 'words']
    return dataframe.set_index('labels')['words'].to_dict()


def get_word(file: str, label_to_word: dict)-> str:
    """

    Args:
        file: file idword_sp if phonebook word/sp if scd
        label_to_word: if None scd else {idword:word}

    Returns:
        readable word
    """
    if label_to_word is None:
        return file.split('/')[-2]
    label = file.split('_')[0]
    return label_to_word[label]


def separate_file_embedding(all_embedding: list)-> list:
    """

    Args:
        all_embedding: [(file, embedding)]

    Returns:
        [[files], [embeddings]]
    """

    return list(map(list, zip(*all_embedding)))


def choose_what_to_keep(problem: str, label_to_word: dict, all_embedding, n)-> list:
    if problem == 'words':
        counter = Counter((get_word(file, label_to_word) for file, _ in all_embedding))
    else:
        counter = Counter((file.split('_')[-1] for file, _ in all_embedding))
    # for lulz
    return sorted(list(map(set, zip(*counter.most_common(n))))[0])


def cos_dist(out1, out2):
    return 1 - F.cosine_similarity(out1, out2, dim=0)


def l2_dist(out1, out2):
    return F.pairwise_distance(out1.unsqueeze(0), out2.unsqueeze(0))


def _choose_negatif(word_embedding):

    def flatten(list_list):
        return (x for subs in list_list for x in subs)
    res = []
    for i, same in enumerate(word_embedding):
        res += [(x, y) for x in same for y in flatten(word_embedding[(i + 1):])]

    return res


def choose_pair(word_embedding: dict, keep_neg: int) -> tuple:
    """

    Args:
        word_embedding: {words:[embeddings]}
        keep_neg: how many example to keep to constitute the negatives pairs

    Returns:
        positive_pairs, negative_pairs
    """
    pos_pair = [x for subs in (combinations(x, 2) for x in word_embedding.values()) for x in subs]
    neg_embedding = [y[:keep_neg] for y in word_embedding.values()]

    neg_pair = _choose_negatif(neg_embedding)
    return pos_pair, neg_pair
