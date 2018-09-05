import torch
import torch.nn as nn
import torch.nn.functional as F


class TripletLossCos(nn.Module):
    def __init__(self, margin):
        assert 0 < margin < 1
        super(TripletLossCos, self).__init__()

        self.margin = margin

    def forward(self, xa, xp, xn):
        return torch.mean(
            torch.clamp(1 - F.cosine_similarity(xa, xp, dim=1) - (1 - F.cosine_similarity(xa, xn, dim=1)) + self.margin,
                        min=0.0))


class LossL2(nn.Module):
    def __init__(self, device):

        super(LossL2, self).__init__()
        self.device = device

    def forward(self, ya, yp, yn):
        net_plus = F.pairwise_distance(ya, yp).unsqueeze(1)
        net_minus = F.pairwise_distance(ya, yn).unsqueeze(1)
        y_final3 = torch.cat([net_plus, net_minus], dim=1)
        label2 = torch.ones(ya.size(0))
        label2 = label2.to(self.device)

        return F.cross_entropy(y_final3, label2.long())


class ContrastiveLoss(torch.nn.Module):
    """
    Contrastive loss function.
    Based on: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    """

    def __init__(self, margin):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2)
        loss_contrastive = torch.mean(label * torch.pow(euclidean_distance, 2) +
                                      (1 - label) * torch.pow(torch.clamp(self.margin -
                                                                          euclidean_distance, min=0.0), 2))

        return loss_contrastive
