import logging
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.metrics import get_auc_pr_threshold
from models.utils import AverageMeter


class SiameseCNN(nn.Module):
    def __init__(self, nb_features, device, embedding_dim, dropout,  problem, non_linearity, margin,
                 normalize, show_every=1500):
        super(SiameseCNN, self).__init__()
        self.normalize = normalize
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
        if self.normalize:
            out_fc = F.normalize(out_fc, p=2)
        return out_fc

    def forward(self, x1, x2):
        out1w = self.forward_once(x1)
        out2w = self.forward_once(x2)
        return out1w, out2w

    def trainer(self, train_loader, criterion, optimizer, epoch, cum_epochs):
        logging.info('\n' + '-' * 200 + '\n' + '\t' * 10 + 'TRAINING\n')
        losses = AverageMeter()
        self.train()
        for i, (x1, x2, label) in enumerate(train_loader):
            x1, x2 = x1.unsqueeze(1), x2.unsqueeze(1)
            if not self.normalize:
                label = 2 * label - 1
            x1, x2, label = x1.to(self.device), x2.to(self.device), label.to(self.device)
            out1, out2 = self.forward(x1, x2)
            optimizer.zero_grad()
            loss = criterion(out1.squeeze(), out2.squeeze(), label.squeeze().float())
            losses.update(loss.item(), x1.size(0))
            loss.backward()
            optimizer.step()
            if (i+1) % self.show_every == 0:
                logging.info('\tEpoch: [{0}][{1}/{2}]\t'
                             'Loss {loss.val:.4f} ({loss.avg:.4f})'.format(
                              cum_epochs + epoch + 1, i+1, len(train_loader), loss=losses))

        logging.info('\n' + '-' * 200)

    def validate(self, val_loader):
        logging.info('\n' + '*' * 200 + '\n' + '\t' * 10 + 'VALIDATION\n')
        assert self.problem in {'words', 'speakers'}
        self.eval()
        y_scores = []
        y_true = []
        with torch.no_grad():
            for i, (x1, x2, labels) in enumerate(val_loader):
                x1, x2 = x1.unsqueeze(1), x2.unsqueeze(1)
                x1, x2, labels = x1.to(self.device), x2.to(self.device), labels.to(self.device)
                out1, out2 = self.forward(x1, x2)
                out = F.cosine_similarity(out1.squeeze(), out2.squeeze())
                y_scores.append(out.cpu())
                y_true.append(labels.cpu())

        acc = get_auc_pr_threshold(y_true, y_scores, self.problem, reverse_pred=False)
        logging.info('\n' + '*' * 200)
        return acc
