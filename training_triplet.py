import time

import argparse
import configparser
import logging
import os
import torch.nn as nn
from torch.utils.data import DataLoader

from data.phonebook_dataloader import DataTriplet, torch, Data, DataTripletRandomPad
from models.loss import LossL2, TripletLossCos
from models.triplet_network import TripletCNN, F
from models.utils import load_checkpoint, save_checkpoint

config2 = configparser.ConfigParser()
config2.read('configuration_files/triplet_phonebook.ini')

parser = argparse.ArgumentParser()
parser.add_argument("config", help="choose configuration version",
                    choices=list(config2.keys())[1:])

parser.add_argument("problem", help="words or speakers", choices=['words', 'speakers'])

parser.add_argument("rd", help="random or centered", choices=['random', 'centered'])

parser.add_argument("file", help="file to save/load model")

parser.add_argument("path", help="where are the shelves files")

parser.add_argument("where_to_save", help="directory to save file")


if __name__ == '__main__':

    extension = '.pth.tar'

    args = parser.parse_args()

    problem = args.problem

    RANDOM = (args.rd == 'random')

    my_version = args.config
    
    where_to_save = args.where_to_save

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    batch_size = int(config2["DEFAULT"]['batch_size'])
    epochs = int(config2["DEFAULT"]['epochs'])
    embedding_dim = int(config2["DEFAULT"]['embedding_dim'])
    dropout = float(config2["DEFAULT"]['dropout'])
    train_file = config2["DEFAULT"]['train']
    val_file = config2["DEFAULT"]['val']
    load_file = config2["DEFAULT"]['preprocess']
    margin = float(config2[my_version]['margin'])
    criterion = config2[my_version]['criterion']

    val_file = val_file.replace('.csv', '_{}.csv'.format(problem))

    try:
        nb_examples = int(config2[my_version]['nb_examples'])
    except KeyError:
        nb_examples = 500

    if "new" in train_file:
        nb_features = 90
        args.file = args.file + "_new"
    else:
        nb_features = 39

    if problem == 'speakers':
        train_file = train_file.replace('words', 'speakers')

    args.file += "_{}_dim_{}_{}_random_{}".format(problem, embedding_dim, criterion, RANDOM)
    exist = [x for x in os.listdir(where_to_save) if x.startswith(args.file) and x.endswith('.log')]
    if len(exist):
        args.file = exist[0][:-4]

    else:
        date = time.strftime("%d_%m_%Y")
        args.file += '_{}'.format(date)

    if criterion == "TripletLoss":
        criterion = TripletLossCos(margin=margin)
        update = True
        reverse_pred = False
        dist_fun = F.cosine_similarity
        kind_dist = 'cos'

    elif criterion == "Loss":
        criterion = LossL2(device=device)
        update = False
        reverse_pred = True
        dist_fun = F.pairwise_distance
        kind_dist = 'L2'

    else:
        raise AttributeError("{} is not a supported criterion".format(criterion))

    logging.basicConfig(filename='{}.log'.format(os.path.join(where_to_save, args.file)), level=logging.INFO,
                        format='%(asctime)s %(message)s', datefmt='%d/%m/%Y %H:%M:%S')

    args.file = args.file + extension

    if RANDOM:
        data_train = DataTripletRandomPad(mod="train", problem=problem, load_file=load_file,
                                          all_words=train_file, nb_examples=batch_size * nb_examples)
    else:
        data_train = DataTriplet(mod="train", problem=problem, load_file=load_file, all_words=train_file,
                                 nb_examples=batch_size * nb_examples)
    train_loader = DataLoader(data_train, shuffle=True, batch_size=batch_size, pin_memory=True)

    data_val = Data(mod="val", data_file=val_file, load_file=load_file, problem=problem, is_8k=False)
    val_loader = DataLoader(data_val, shuffle=False, batch_size=batch_size, pin_memory=True)

    non_linearity = getattr(nn, config2[my_version]['non_linearity'])()
    model = TripletCNN(nb_features=nb_features, device=device, embedding_dim=embedding_dim, dropout=dropout,
                       problem=problem, non_linearity=non_linearity, margin=margin,
                       show_every=max(1, nb_examples//2), kind_dist=kind_dist).to(device)
    logging.info(model)
    optimizer = torch.optim.Adadelta(model.parameters())

    try:
        best_prec1, cum_epochs, model, optimizer = load_checkpoint(where_to_save, args.file,
                                                                   model, optimizer)
    except FileNotFoundError:
        cum_epochs = 0
        best_prec1 = 0
        pass

    for epoch in range(epochs):
        # train for one epoch
        model.trainer(train_loader, criterion, optimizer, epoch,
                      cum_epochs, update=update)
        # evaluate on validation set
        auc_score = model.validate(val_loader=val_loader, dist_fun=dist_fun, reverse_pred=reverse_pred)
        is_best = auc_score > best_prec1
        best_prec1 = max(auc_score, best_prec1)
        save_checkpoint(
            {
                'epoch': epoch + cum_epochs + 1,
                'arch': str(model),
                'state_dict': model.state_dict(),
                'best_prec1': best_prec1,
                'optimizer': optimizer.state_dict(),
            }, where_to_save, args.file, is_best)
