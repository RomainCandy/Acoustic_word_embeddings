import argparse
import configparser
import logging
import os
import time

import torch.nn as nn
from torch.utils.data import DataLoader

from data.phonebook_dataloader import DataSiamese, torch
from models.loss import ContrastiveLoss
from models.siamese_network import SiameseCNN
from models.utils import load_checkpoint, save_checkpoint

config2 = configparser.ConfigParser()
config2.read('configuration_files/siamese_phonebook.ini')

parser = argparse.ArgumentParser()
parser.add_argument("config", help="choose configuration version",
                    choices=list(config2.keys())[1:])

parser.add_argument("problem", help="words or speakers", choices=['words', 'speakers'])

parser.add_argument("file", help="file to save/load model")

parser.add_argument("path", help="where are the shelves prepossessing files")

parser.add_argument("where_to_save", help="directory to save file")


if __name__ == '__main__':

    extension = '.pth.tar'

    args = parser.parse_args()

    problem = args.problem

    my_version = args.config
    where_to_save = args.where_to_save
    path = args.path

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    batch_size = int(config2['DEFAULT']['batch_size'])
    epochs = int(config2['DEFAULT']['epochs'])
    dropout = float(config2['DEFAULT']['dropout'])
    load_file = config2['DEFAULT']['preprocess']
    margin = float(config2[my_version]['margin'])
    criterion = config2[my_version]['criterion']
    non_linearity = getattr(nn, config2[my_version]['non_linearity'])()
    embedding_dim = int(config2[my_version]['embedding_dim'])

    train_file = next((os.path.join(path, x) for x in os.listdir(path)
                       if x.startswith('{}_train'.format(problem))), AttributeError('no file found'))

    val_file = next((os.path.join(path, x) for x in os.listdir(path)
                     if x.startswith('{}_val'.format(problem))), AttributeError('no file found'))

    if criterion == 'ContrastiveLoss':
        criterion = ContrastiveLoss(margin=margin)
        normalize = True

    else:
        criterion = getattr(nn, criterion)(margin)
        normalize = False

    criterion.margin = margin

    args.file += "_{}_dim_{}_{}".format(problem, embedding_dim, type(criterion).__name__)
    exist = [x for x in os.listdir(where_to_save) if x.startswith(args.file) and x.endswith('.log')]
    if len(exist):
        args.file = exist[0][:-4]

    else:
        date = time.strftime("%d_%m_%Y")
        args.file += '_{}'.format(date)

    logging.basicConfig(filename='{}.log'.format(os.path.join(where_to_save, args.file)), level=logging.INFO,
                        format='%(asctime)s %(message)s', datefmt='%d/%m/%Y %H:%M:%S')

    args.file = args.file + extension

    data_train = DataSiamese(mod="train", data_file=train_file, load_file=load_file, problem=problem)
    train_loader = DataLoader(data_train, shuffle=True, batch_size=batch_size, pin_memory=True)

    data_val = DataSiamese(mod="val", data_file=val_file, load_file=load_file, problem=problem)
    val_loader = DataLoader(data_val, shuffle=False, batch_size=batch_size, pin_memory=True)

    model = SiameseCNN(nb_features=39, device=device, embedding_dim=embedding_dim, dropout=dropout,
                       problem=problem, non_linearity=non_linearity, margin=margin, normalize=normalize,
                       show_every=1000).to(device)

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
                      cum_epochs)
        # evaluate on validation set
        auc_score = model.validate(val_loader=val_loader)
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
