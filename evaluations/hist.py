import time

import matplotlib.pyplot as plt
import os
import torch

from .load_models import get_all, load_model, get_all_embedding, word_to_embedding
from .plots import plot_hist
from .utils import l2_dist, cos_dist, choose_pair, create_directory


def main_hist(mod, problem, config_file, version, file_words, triplet, lab_to_word, path_to_save, n_keep_neg=2):
    assert n_keep_neg < 20

    device = torch.device("cpu")

    test_loader, label_to_word, config = get_all(mod=mod, problem=problem,
                                                 config_file=config_file, lab_to_word=lab_to_word)

    date = time.strftime("%d_%m_%Y")

    if path_to_save:

        create_directory(path_to_save)

    loss = config[version]['criterion']
    if loss == 'Loss' or loss == 'ContrastiveLoss':
        dist_fun = l2_dist

    elif loss == 'TripletLoss' or loss == 'CosineEmbeddingLoss':
        dist_fun = cos_dist

    else:
        raise AttributeError

    model_words = load_model(config, version, file_words, device, triplet=triplet)

    all_embedding = get_all_embedding(model_words, test_loader)

    res = word_to_embedding(all_embedding, label_to_word=label_to_word, problem=problem)

    pos_pair, neg_pair = choose_pair(word_embedding=res, keep_neg=n_keep_neg)

    dist_pos = [dist_fun(x1, x2) for x1, x2 in pos_pair]
    dist_neg = [dist_fun(x1, x2) for x1, x2 in neg_pair]

    fig = plt.figure(1, figsize=(6, 6))

    plot_hist(dist_pos, dist_neg,
              title='histogram of the {0:} between '
                    'same {1:} \nand different {1:}'.format(dist_fun.__name__, problem))

    if path_to_save:

        fig.savefig(os.path.join(path_to_save, '{}_{}_{}_{}_{}_{}_{}.png'.format(date, version, mod,
                                                                                 dist_fun.__name__,
                                                                                 len(pos_pair), len(neg_pair),
                                                                                 n_keep_neg)))
    else:
        plt.show()
