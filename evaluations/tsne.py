import time

import matplotlib.pyplot as plt
import numpy as np
import os
import torch
from sklearn import decomposition
from sklearn import manifold
from sklearn.pipeline import Pipeline

from .load_models import get_all, load_model, get_all_embedding
from .plots import projection
from .utils import separate_file_embedding, choose_what_to_keep, get_word, create_directory


def pca_tsne(dist_fun: str)-> Pipeline:
    """

    Args:
        dist_fun: see manifold.TSNE doc

    Returns:
        pipeline PCA(50) -> TSNE
    """
    pca = decomposition.PCA(n_components=50)
    tsne = manifold.TSNE(n_components=2, init='pca', random_state=0, metric=dist_fun)
    pipe = Pipeline([('pca', pca), ('tsne', tsne)])
    return pipe


def main_tsne(mod, problem, config_file, version, file_words, triplet, lab_to_word, path_to_save, n_keep):

    device = torch.device("cpu")

    dict_models, test_loader, label_to_word, config = get_all(mod=mod, problem=problem,
                                                              config_file=config_file, lab_to_word=lab_to_word)
    date = time.strftime("%d_%m_%Y")

    if path_to_save:
        create_directory(path_to_save)

    loss = config[version]['criterion']
    if loss == 'Loss' or loss == "ContrastiveLoss":
        dist_fun = 'euclidean'

    elif loss == 'TripletLoss' or loss == 'CosineEmbeddingLoss':
        dist_fun = 'cosine'

    else:
        raise AttributeError(loss)

    model_words = load_model(config, version, file_words, device, triplet=triplet)

    all_embedding = get_all_embedding(model_words, test_loader)

    labels, data = separate_file_embedding(all_embedding)

    data = [x.numpy() for x in data]

    if problem == 'words':

        labels = np.array([get_word(f, label_to_word) for f in labels])
    else:
        labels = np.array([f.split('_')[-1] for f in labels])
    keep_test = choose_what_to_keep(problem=problem, label_to_word=label_to_word,
                                    all_embedding=all_embedding, n=n_keep)

    data, labels = zip(*((x, y) for x, y in zip(data, labels) if y in keep_test))
    labels = np.array(labels)

    algo1 = manifold.TSNE(n_components=2, init='pca', random_state=0, metric=dist_fun)
    algo = [algo1]
    for al in algo:
        fig, ax = plt.subplots(figsize=(8, 8))
        projection(al, data, labels, keep_test,
                   title='{} {} {}'.format(type(al).__name__, mod, version), ax=ax)

        fig.legend(loc=7)
        fig.tight_layout()
        fig.subplots_adjust(right=0.75)
        if path_to_save:
            fig.savefig(
                os.path.join(path_to_save, '{}_{}_{}_{}_{}_{}.png'.format(date, version, mod, dist_fun,
                                                                          type(al).__name__,
                                                                          n_keep)))
        else:
            plt.show()

        plt.close(plt.gcf())
