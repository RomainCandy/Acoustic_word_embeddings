import logging
import os
import shutil
import torch
import torch.nn.functional as F


class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self):
        self.val = None
        self.avg = None
        self.sum = None
        self.count = None
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def save_checkpoint(state, where_to_save, filename, is_best=False):
    torch.save(state, os.path.join(where_to_save, filename))
    if is_best:
        logging.info('\tnew best!')
        shutil.copyfile(os.path.join(where_to_save, filename), os.path.join(where_to_save, 'best_' + filename))


def load_checkpoint(where_to_save, filename, model, optimizer):
    file = os.path.join(where_to_save, filename)
    if os.path.isfile(file):
        logging.info("=> loading checkpoint '{}'".format(filename))
        checkpoint = torch.load(file, map_location=lambda storage, loc: storage)
        logging.info("arch: {}".format(checkpoint['arch']))
        new_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        best_prec1 = checkpoint['best_prec1']
        logging.info("=> loaded checkpoint '{}' (epoch {})".format(filename, checkpoint['epoch']))
        logging.info("precision so far on validation set :{:.4f}".format(checkpoint['best_prec1']))
        return best_prec1, new_epoch, model, optimizer
    raise FileNotFoundError('{} is not here'.format(file))


def dcos(xa, xp):
    return 1 - F.cosine_similarity(xa, xp)


def update_cos(ya, yp, yn, margin):
    return F.cosine_similarity(ya, yn) * (dcos(ya, yn) < dcos(ya, yp) + margin).float()
