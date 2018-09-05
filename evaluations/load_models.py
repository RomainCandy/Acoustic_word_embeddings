import configparser
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data import DataAll
from models import SiameseCNN, TripletCNN
from .utils import get_label_to_word, get_word


def load_model(config: configparser.ConfigParser, version: str, file_words, device, normalize=False, triplet=True):
    """
    Args:
        config: the configuration files
        version: which version to use in config
        file_words: saved model (torch.save)
        device: cpu or gpu
        normalize: if True L2 normalization
        triplet: if False Siamese

    Returns:

    """
    embedding_dim = int(config[version]['embedding_dim'])
    dropout = float(config[version]['dropout'])
    margin = float(config[version]['margin'])
    non_linearity = getattr(nn, config[version]['non_linearity'])()
    criterion = config[version]['criterion']

    if criterion == 'TripletLoss' or criterion == 'CosineEmbeddingLoss':
        kind_dist = 'cos'
        normalize = False

    elif criterion == 'Loss' or criterion == 'ContrastiveLoss':
        kind_dist = 'L2'
        normalize = normalize

    else:
        raise AttributeError('{} is wrong'.format(criterion))

    nb_features = 39

    if triplet:
        model_words = TripletCNN(nb_features=nb_features, device=device, embedding_dim=embedding_dim,
                                 dropout=dropout, problem='words', non_linearity=non_linearity,
                                 margin=margin, kind_dist=kind_dist)
    else:
        model_words = SiameseCNN(nb_features=nb_features, device=device, embedding_dim=embedding_dim,
                                 dropout=dropout, problem='words', non_linearity=non_linearity,
                                 margin=margin, normalize=normalize)

    checkpoint_words = torch.load(file_words, map_location=lambda storage, loc: storage)

    model_words.load_state_dict(checkpoint_words['state_dict'])

    return model_words


def get_loader(config: configparser.ConfigParser, version: str, mod: str, problem: str):
    """

    Args:
        config: the configuration files
        version: which version to use in config
        mod: train, val or test sets
        problem: words or speakers

    Returns:

    """
    assert mod in {"train", "val", "test"}
    batch_size = int(config[version]['batch_size'])

    load_file = config[version]['preprocess']

    data_test = DataAll(mod=mod, problem=problem, load_file=load_file)
    test_loader = DataLoader(data_test, shuffle=False, batch_size=batch_size)

    return test_loader


def get_all_embedding(model_words, loader)-> list:

    """
    Args:
        model_words: a torch nn.Module
        loader: DataLoader

    Returns:
        [(file,embedding)]
    """
    model_words.eval()
    all_embedding = []
    with torch.no_grad():
        for i, (x, file) in enumerate(loader):
            x = x.unsqueeze(1)
            out_words = model_words.forward_once(x)
            all_embedding.extend([(f, emb) for f, emb in zip(file, out_words)])
    return all_embedding


def get_all(mod, problem, config_file, lab_to_word):

    config = configparser.ConfigParser()
    config.read(config_file)

    test_loader = get_loader(config, 'DEFAULT', mod=mod, problem=problem)

    label_to_word = get_label_to_word(lab_to_word)
    return test_loader, label_to_word, config


def word_to_embedding(list_file_embeddings: list, label_to_word: dict, problem: str)-> dict:
    """

    Args:
        list_file_embeddings: [(files, embeddings)]
        label_to_word: if None scd else {idword:word}
        problem : words or speakers

    Returns:
        {word:[embedding]}
    """
    res = {}
    for id_word, embedding in list_file_embeddings:
        if problem == 'words':
            word = get_word(id_word, label_to_word)
        else:
            word = id_word.split('_')[-1]
        try:
            res[word].append(embedding)
        except KeyError:
            res[word] = [embedding]
    return res
