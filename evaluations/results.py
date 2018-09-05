import time

import matplotlib.pyplot as plt
import os
import torch
from sklearn.metrics import f1_score, accuracy_score

from .load_models import get_all, load_model, get_all_embedding, word_to_embedding
from .plots import np, plot_ap, plot_auc, plot_confusion_matrix
from .utils import l2_dist, cos_dist, choose_pair, create_directory


def main_f1_ap(mod, problem, config_file, version, file_words, triplet, lab_to_word,
               path_to_save, margin, n_keep_neg=2):

    assert n_keep_neg < 20

    device = torch.device("cpu")

    dict_models, test_loader, label_to_word, config = get_all(mod=mod, problem=problem,
                                                              config_file=config_file, lab_to_word=lab_to_word)

    date = time.strftime("%d_%m_%Y")

    if path_to_save:
        create_directory(path_to_save)

    loss = config[version]['criterion']
    if loss == 'Loss' or loss == 'ContrastiveLoss':
        dist_fun = l2_dist
        kind_dist = 'L2'

    elif loss == 'TripletLoss' or loss == 'CosineEmbeddingLoss':
        dist_fun = cos_dist
        kind_dist = 'cos'

    else:
        raise AttributeError

    model_words = load_model(config, version, file_words, device, triplet=triplet)

    all_embedding = get_all_embedding(model_words, test_loader)

    res = word_to_embedding(all_embedding, label_to_word=label_to_word, problem=problem)

    pos_pair, neg_pair = choose_pair(word_embedding=res, keep_neg=n_keep_neg)

    dist_pos = [-1*dist_fun(x1, x2) for x1, x2 in pos_pair]
    dist_neg = [-1*dist_fun(x1, x2) for x1, x2 in neg_pair]

    scores = np.array(dist_pos + dist_neg)

    labels = np.ones(shape=len(scores))
    labels[len(dist_pos):] = 0

    pred = scores > -1 * margin

    acc = accuracy_score(labels, pred)

    f1 = f1_score(labels, pred)

    fig, ax = plt.subplots(2, 2, figsize=(15, 10), squeeze=False)

    plot_auc(y_true=labels, y_score=scores, title='AUC', ax=ax[0, 0])

    plot_ap(y_true=labels, y_score=scores, title='AP', ax=ax[1, 0])

    plot_confusion_matrix(y_true=labels, y_pred=pred, classes=['DW', 'SW'], normalize=False, ax=ax[0, 1])

    plot_confusion_matrix(y_true=labels, y_pred=pred, classes=['DW', 'SW'], normalize=True, ax=ax[1, 1])

    if path_to_save:

        fig.savefig(os.path.join(path_to_save, '{}_{}_{}_{}_acc_{:.4f}_f1_{:.4f}'
                                               '_keep_{}_{}_{}.png'.format(date, kind_dist, version, mod, acc,
                                                                           f1, n_keep_neg,
                                                                           len(dist_pos), len(dist_neg))))
    else:
        plt.show()
    plt.close(plt.gcf())
