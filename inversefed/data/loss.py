"""Define various loss functions and bundle them with appropriate metrics."""

import torch
import numpy as np
import torch.nn as nn

class Loss:
    """Abstract class, containing necessary methods.

    Abstract class to collect information about the 'higher-level' loss function, used to train an energy-based model
    containing the evaluation of the loss function, its gradients w.r.t. to first and second argument and evaluations
    of the actual metric that is targeted.

    """

    def __init__(self):
        """Init."""
        pass

    def __call__(self, reference, argmin):
        """Return l(x, y)."""
        raise NotImplementedError()
        return value, name, format

    def metric(self, reference, argmin):
        """The actually sought metric."""
        raise NotImplementedError()
        return value, name, format


class PSNR(Loss):
    """A classical MSE target.

    The minimized criterion is MSE Loss, the actual metric is average PSNR.
    """

    def __init__(self):
        """Init with torch MSE."""
        self.loss_fn = torch.nn.MSELoss(size_average=None, reduce=None, reduction='mean')

    def __call__(self, x=None, y=None):
        """Return l(x, y)."""
        name = 'MSE'
        format = '.6f'
        if x is None:
            return name, format
        else:
            value = 0.5 * self.loss_fn(x, y)
            return value, name, format

    def metric(self, x=None, y=None):
        """The actually sought metric."""
        name = 'avg PSNR'
        format = '.3f'
        if x is None:
            return name, format
        else:
            value = self.psnr_compute(x, y)
            return value, name, format


    @staticmethod
    def psnr_compute(img_batch, ref_batch, batched=False, factor=1.0):
        """Standard PSNR."""
        def get_psnr(img_in, img_ref):
            mse = ((img_in - img_ref)**2).mean()
            if mse > 0 and torch.isfinite(mse):
                return (10 * torch.log10(factor**2 / mse)).item()
            elif not torch.isfinite(mse):
                return float('nan')
            else:
                return float('inf')

        if batched:
            psnr = get_psnr(img_batch.detach(), ref_batch)
        else:
            [B, C, m, n] = img_batch.shape
            psnrs = []
            for sample in range(B):
                psnrs.append(get_psnr(img_batch.detach()[sample, :, :, :], ref_batch[sample, :, :, :]))
            psnr = np.mean(psnrs)

        return psnr


class Classification(Loss):
    """A classical NLL loss for classification. Evaluation has the softmax baked in.

    The minimized criterion is cross entropy, the actual metric is total accuracy.
    """

    def __init__(self):
        """Init with torch MSE."""
        self.loss_fn = torch.nn.CrossEntropyLoss(weight=None, size_average=None, ignore_index=-100,
                                                 reduce=None, reduction='mean')

    def __call__(self, x=None, y=None):
        """Return l(x, y)."""
        name = 'CrossEntropy'
        format = '1.5f'
        # if x is None:
        #     return name, format
        # else:
        #     value = 0.5 * self.loss_fn(x, y)
        #     return value, name, format

        if isinstance(x, tuple):
            value = 0.5 * self.loss_fn(x[0], y)
        else:
            value = 0.5 * self.loss_fn(x, y)
            
        return value
            

    def metric(self, x=None, y=None):
        """The actually sought metric."""
        name = 'Accuracy'
        format = '6.2%'
        if x is None:
            return name, format
        else:
            value = (x.data.argmax(dim=1) == y).sum().float() / y.shape[0]
            return value.detach(), name, format


# class TripletLoss(nn.Module):
#     def __init__(self, margin=0.2):
#         super(TripletLoss, self).__init__()
#         self.margin = margin
#
#     def forward(self, anchor, positive, negative):
#         distance_pos = torch.sqrt(torch.sum(torch.pow(anchor - positive, 2), dim=1))
#         distance_neg = torch.sqrt(torch.sum(torch.pow(anchor - negative, 2), dim=1))
#         losses = torch.relu(distance_pos - distance_neg + self.margin)
#         loss = torch.mean(losses)
#         return loss




class TripletLoss(nn.Module):
    def __init__(self, margin=1):
        super(TripletLoss, self).__init__()
        self.margin = margin

    # def forward(self, anchor, positive, negative):
    #     # distance_pos = torch.sqrt(torch.sum(torch.pow(anchor - positive, 2), dim=1))
    #     # distance_neg = torch.sqrt(torch.sum(torch.pow(anchor - negative, 2), dim=1))
    #     # losses = torch.relu(distance_pos - distance_neg + self.margin)
    #     # loss = torch.mean(losses)
    #     triplet_loss = nn.TripletMarginLoss(margin=1.0, p=2)
    #     loss = triplet_loss(anchor, positive, negative)
    #
    #     return loss

    def forward(self, anchor, ori, positive, negative, neg_pos):
        # distance_pos = torch.sqrt(torch.sum(torch.pow(anchor - positive, 2), dim=1))
        # distance_neg = torch.sqrt(torch.sum(torch.pow(anchor - negative, 2), dim=1))
        # losses = torch.relu(distance_pos - distance_neg + self.margin)
        # loss = torch.mean(losses)
        triplet_loss = nn.TripletMarginLoss(margin=1.0, p=2)
        loss = triplet_loss(anchor, positive, negative)

        return loss


class TripletLoss_PSNR(nn.Module):
    def __init__(self, margin=1, lambda1 = 0.1, lambda2 =0.1):
        super(TripletLoss_PSNR, self).__init__()
        self.margin = margin
        self.lambda1 = lambda1
        self.lambda2 = lambda2

    def forward(self, anchor, positive1, negative1, positive2, negative2, psnr):
        psnr_pos1 = psnr[:, 1]
        psnr_pos2 =  psnr[:, 2]
        psnr_neg1 =  psnr[:, 3]
        psnr_neg2  =  psnr[:, 4]

        distance_pos1 = torch.sqrt(torch.sum(torch.pow(anchor - positive1, 2), dim=1))
        distance_pos2 = torch.sqrt(torch.sum(torch.pow(anchor - positive2, 2), dim=1))
        distance_neg1 = torch.sqrt(torch.sum(torch.pow(anchor - negative1, 2), dim=1))
        distance_neg2 = torch.sqrt(torch.sum(torch.pow(anchor - negative2, 2), dim=1))


        positive_weight = torch.sigmoid(psnr_pos2 - psnr_pos1)
        negative_weight = torch.sigmoid(psnr_neg2 - psnr_neg1)

        distances = [distance_pos1, distance_pos2, distance_neg1, distance_neg2]
        weights = [positive_weight, 1 - positive_weight, negative_weight, 1 - negative_weight]
        weighted_distances = [distances[i] * weights[i] for i in range(len(distances))]

        losses = torch.relu(weighted_distances[0] - weighted_distances[2] + self.margin) \
                 + torch.relu(weighted_distances[1] - weighted_distances[3] + self.margin) \
                 + torch.relu(weighted_distances[0] - weighted_distances[3] + self.margin) \
                 + torch.relu(weighted_distances[1] - weighted_distances[2] + self.margin)

        loss = torch.mean(losses)
        #
        # loss = torch.relu(weighted_distances[0] - weighted_distances[2] + self.margin) + torch.relu(weighted_distances[1] - weighted_distances[3] + self.margin) \
        #        + 0.1 * (torch.pow(distance_pos1 - distance_pos2, 2) - torch.pow(distance_neg1 - distance_neg2, 2))
        return loss




class LabelSmoothing(nn.Module):
 
    def __init__(self, size, smoothing=0.0):
        super(LabelSmoothing, self).__init__()
        self.criterion = nn.KLDivLoss(size_average=False)
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size
        self.true_dist = None
 
    def forward(self, x, target):
        assert x.size(1) == self.size
        true_dist = x.data.clone()
        true_dist.fill_(self.smoothing / (self.size - 1))
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        self.true_dist = true_dist
        return self.criterion(x, true_dist.detach())