import logging
import numpy as np
import torch
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, roc_curve, classification_report


def get_auc_pr_threshold(y_true, y_score, problem, reverse_pred):

    target_names = ['Diff {}'.format(problem), 'Same {}'.format(problem)]

    y_score = torch.cat(y_score).view(-1)
    y_true = torch.cat(y_true).view(-1)

    if reverse_pred:
        y_true = 1 - y_true
        target_names = ['Same {}'.format(problem), 'Diff {}'.format(problem)]

    auc_score = roc_auc_score(y_true=y_true, y_score=y_score)
    ap_score = average_precision_score(y_true=y_true, y_score=y_score)
    logging.info('\tAUC {}: {:.4f}\t, AP {}: {:.4f}\n'.format(problem, auc_score, problem, ap_score))

    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]
    y_pred = y_score > float(optimal_threshold)
    acc = accuracy_score(y_pred, y_true)

    logging.info('\n{}'.format(
        classification_report(y_pred, y_true, target_names=target_names)))
    f1 = f1_score(y_true, y_pred)
    logging.info('\tOptimal_threshold :{:.4f}, \tAcc on this threshold :{:.4f}, \tF1 on this threshold: {:.4f}'.format(
        optimal_threshold, acc, f1))

    return auc_score
