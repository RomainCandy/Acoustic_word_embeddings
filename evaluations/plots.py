import itertools

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn import manifold
from sklearn.metrics import roc_curve, auc, confusion_matrix, precision_recall_curve, average_precision_score


def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=True,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues, ax=None):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.

    """

    cm = confusion_matrix(y_true, y_pred)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    if ax is None:

        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)

        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], fmt),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')

    else:
        fig = ax.imshow(cm, interpolation='nearest', cmap=cmap)
        ax.set_title(title)
        plt.colorbar(fig, ax=ax)
        tick_marks = np.arange(len(classes))
        ax.set_xticks(tick_marks)
        ax.set_xticklabels(classes, rotation=45)
        ax.set_yticks(tick_marks)
        ax.set_yticklabels(classes)
        ax.set_aspect('auto')

        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            ax.text(j, i, format(cm[i, j], fmt),
                    horizontalalignment="center",
                    color="white" if cm[i, j] > thresh else "black")

        plt.tight_layout()
        ax.set_ylabel('True label')
        ax.set_xlabel('Predicted label')


def plot_ap(y_true, y_score, title, ax=None):

    precision, recall, _ = precision_recall_curve(y_true, y_score)
    average_precision = average_precision_score(y_true, y_score)

    if ax is None:
        plt.step(recall, precision, color='b', alpha=0.2,
                 where='post')
        plt.fill_between(recall, precision, step='post', alpha=0.2,
                         color='b')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.ylim([0.0, 1.05])
        plt.xlim([0.0, 1.0])
        plt.title('{}: AP={:0.2f}'.format(title, average_precision))

    else:
        ax.step(recall, precision, color='b', alpha=0.2,
                where='post')
        ax.fill_between(recall, precision, step='post', alpha=0.2,
                        color='b')
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_ylim([0.0, 1.05])
        ax.set_xlim([0.0, 1.0])
        ax.set_title('{}: AP={:.2f}'.format(title, average_precision))


def plot_auc(y_true, y_score, title, ax=None):

    fpr, tpr, _ = roc_curve(y_true, y_score)

    roc_auc = auc(fpr, tpr)
    lw = 2
    if ax is None:

        plt.plot(fpr, tpr, color='darkorange',
                 lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(title)
        plt.legend(loc="lower right")

    else:
        ax.plot(fpr, tpr, color='darkorange',
                lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
        ax.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title(title)
        ax.legend(loc="lower right")


def plot_hist(pos_dist, neg_dist, title):

    sns.distplot(pos_dist, bins=int(180 / 5),
                 label='positive pairs distances',
                 hist_kws={'edgecolor': 'black'})

    sns.distplot(neg_dist, bins=int(180 / 5),
                 label='negative pairs distances',
                 hist_kws={'edgecolor': 'red'})

    plt.legend(prop={'size': 16})
    plt.title(title)
    plt.xlabel('Distance')
    plt.ylabel('Normalized Counts')


def projection(algo: manifold, data, labels, keep, title, ax):

    """

    Args:
        algo: algorithm to reduce dimension
        data: list or np.array or torch.tensor, the embedding representation
        labels: labels of each point in data
        keep: which labels to keep
        title: title for the plot
        ax: ax for plotting tsne
    Returns:

    """

    y_fit = algo.fit_transform(data)
    scatter_x = y_fit[:, 0]
    scatter_y = y_fit[:, 1]

    num_colors = len(keep)

    colors = plt.cm.gist_ncar(np.linspace(0, 1, num_colors + 2))

    ax.set_prop_cycle('color', colors)

    for g in keep:
        i = np.where(labels == g)
        ax.scatter(scatter_x[i], scatter_y[i], label=g)
        for ii in i[0]:
            ax.annotate(g[0], (scatter_x[ii], scatter_y[ii]))
    ax.set_title(title)
