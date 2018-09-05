import logging
import numpy as np
import torch
import torch.nn as nn

from models.metrics import get_auc_pr_threshold
from models.utils import AverageMeter, update_cos, F


class TripletCNN(nn.Module):
    def __init__(self, nb_features, device, embedding_dim, dropout,  problem, non_linearity, margin,
                 kind_dist, normalize=False, show_every=1500):
        assert kind_dist in {'L2', 'cos'}
        super(TripletCNN, self).__init__()
        self.normalize = normalize
        self.kind_dist = kind_dist
        self.show_every = show_every
        self.embedding_dim = embedding_dim
        self.margin = margin
        self.non_linearity = non_linearity
        self.problem = problem
        self.nb_features = nb_features
        self.dropout = dropout
        self.device = device
        self.features = nn.Sequential(
            nn.Conv2d(1, 96, kernel_size=(9, nb_features)),
            nn.Dropout2d(self.dropout),
            non_linearity,
            nn.MaxPool2d(kernel_size=(3, 1)),
            nn.Conv2d(96, 96 * 2, kernel_size=(8, 1)),
            nn.Dropout2d(self.dropout),
            non_linearity,
            nn.MaxPool2d(kernel_size=(3, 1)),
        )
        self.fc = nn.Sequential(
            nn.Linear(96 * 2 * 30, 96 * 2 * 30),
            nn.Dropout(self.dropout),
            non_linearity,

            nn.Linear(96 * 2 * 30, self.embedding_dim),
        )

    def forward_once(self, x):
        out_conv = self.features(x)
        out_fc = self.fc(out_conv.view(-1, 96 * 2 * 30))
        if self.kind_dist == "L2" and self.normalize:
            out_fc = F.normalize(out_fc, p=2)
        return out_fc

    def forward(self, x1, x2, x3):
        out1w = self.forward_once(x1)
        out2w = self.forward_once(x2)
        out3w = self.forward_once(x3)
        return out1w, out2w, out3w

    def trainer(self, train_loader, criterion, optimizer, epoch, cum_epochs, update):
        logging.info('\n' + '-' * 200 + '\n' + '\t' * 10 + 'TRAINING\n')
        losses = AverageMeter()
        self.train()
        for i, ((ida, xa, xp), (idn, xn)) in enumerate(train_loader):
            xa, xp, xn = xa.unsqueeze(1), xp.unsqueeze(1), xn.unsqueeze(1)
            xa, xp, xn = xa.to(self.device), xp.to(self.device), xn.to(self.device)
            outa, outp, outn = self.forward(xa, xp, xn)
            optimizer.zero_grad()
            loss = criterion(outa, outp, outn)
            losses.update(loss.item(), xa.size(0))
            loss.backward()
            optimizer.step()

            if (i+1) % self.show_every == 0:
                logging.info('\tEpoch: [{0}][{1}/{2}]\t'
                             'Loss {loss.val:.4f} ({loss.avg:.4f})'.format(
                              cum_epochs + epoch + 1, i+1, len(train_loader), loss=losses))
            if update:
                score_update = update_cos(outa, outp, outn, 1.5*self.margin)
                train_loader.dataset.update(score_update, ida, idn)
        if update:
            logging.info('\n{}'.format(np.array_str(train_loader.dataset.probability_matrix,
                                                    precision=2, suppress_small=True)))
            train_loader.dataset.reset()

        logging.info('\n' + '-' * 200)

    def validate(self, val_loader, dist_fun, reverse_pred):
        logging.info('\n' + '*' * 200 + '\n' + '\t' * 10 + 'VALIDATION\n')
        assert self.problem in {'words', 'speakers'}
        self.eval()
        y_scores = []
        y_true = []
        with torch.no_grad():
            for i, (xa, xp, xn) in enumerate(val_loader):
                xa, xp, xn = xa.unsqueeze(1), xp.unsqueeze(1), xn.unsqueeze(1)
                xa, xp, xn = xa.to(self.device), xp.to(self.device), xn.to(self.device)
                outa, outp, outn = self.forward(xa, xp, xn)
                out1 = dist_fun(outa, outp)
                out2 = dist_fun(outa, outn)
                y_scores.append(out1.cpu())
                y_scores.append(out2.cpu())
                y_true.append(torch.ones(xa.size(0), 1))
                y_true.append(torch.zeros(xa.size(0), 1))
        auc_score = get_auc_pr_threshold(y_true, y_scores, self.problem, reverse_pred)
        logging.info('\n' + '*' * 200)
        return auc_score
