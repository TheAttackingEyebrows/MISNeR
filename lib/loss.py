import torch
from torch.autograd import Function
import numpy as np
import torch.nn.functional as F
class DiceCoeff(Function):
    """Dice coeff for individual examples"""

    def forward(self, input, target):
        self.save_for_backward(input, target)
        eps = 0.0001
        self.inter = torch.dot(input.view(-1), target.view(-1))
        self.union = torch.sum(input) + torch.sum(target) + eps

        t = (2 * self.inter.float() + eps) / self.union.float()
        return t

    # This function has only a single output, so it gets only one gradient
    def backward(self, grad_output):

        input, target = self.saved_variables
        grad_input = grad_target = None

        if self.needs_input_grad[0]:
            grad_input = grad_output * 2 * (target * self.union - self.inter) \
                         / (self.union * self.union)
        if self.needs_input_grad[1]:
            grad_target = None

        return grad_input, grad_target


def dice_coeff(input, target):
    """Dice coeff for batches"""
    if input.is_cuda:
        s = torch.FloatTensor(1).cuda().zero_()
    else:
        s = torch.FloatTensor(1).zero_()

    for i, c in enumerate(zip(input, target)):
        s = s + DiceCoeff().forward(c[0], c[1])

    return s / (i + 1)

def jaccard_index(target, pred, num_classes):
    ious = []
    pred = pred.view(-1)
    target = target.view(-1)

    # Ignore IoU for background class ("0")
    for cls in range(1, num_classes):  # This goes from 1:n_classes-1 -> class "0" is ignored
        pred_inds = pred == cls
        target_inds = target == cls

        intersection = pred_inds[target_inds].long().sum().data.cpu()  # Cast to long to prevent overflows
        union = pred_inds.long().sum().data.cpu() + target_inds.long().sum().data.cpu() - intersection
        if union == 0:
            ious.append(float('nan'))  # If there is no ground truth, do not include in evaluation
        else:
            ious.append(float(intersection) / float(union))
    return np.array(ious)

def chamfer_directed(A, B):


    N1 = A.shape[1]
    N2 = B.shape[1]

    if N1 > 0 and N2 > 0:
        y1 = A[:, :, None].repeat(1, 1, N2, 1)
        y2 = B[:, None].repeat(1, N1, 1, 1)

        diff = torch.sum((y1 - y2) ** 2, dim=3)

        loss, _ = torch.min(diff, dim=2)

        loss = torch.mean(loss)
    else:
        loss = torch.Tensor([float("Inf")]).cuda() if A.is_cuda else torch.Tensor([float("Inf")])

    return loss

def chamfer_symmetric(A, B):
    N1 = A.shape[1]
    N2 = B.shape[1]
    y1 = A[:, :, None].repeat(1, 1, N2, 1)
    y2 = B[:, None].repeat(1, N1, 1, 1)

    diff = torch.sum((y1 - y2) ** 2, dim=3)

    loss1, _ = torch.min(diff, dim=1)
    loss2, _ = torch.min(diff, dim=2)

    loss = torch.sum(loss1) + torch.sum(loss2)
    return loss

def chamfer_weighted_symmetric(A, B):

    N1 = A.shape[1]
    N2 = B.shape[1]
    y1 = A[:, :, None].repeat(1, 1, N2, 1)
    y2 = B[:, None].repeat(1, N1, 1, 1)

    diff = torch.sum((y1 - y2) ** 2, dim=3)

    loss1, _ = torch.min(diff, dim=1)
    loss2, _ = torch.min(diff, dim=2)
    loss = torch.mean(loss1) + torch.mean(loss2)
    return loss

def chamfer_weighted_symmetric_with_dtf(A, B, B_dtf):

    N1 = A.shape[1]
    N2 = B.shape[1]
    y1 = A[:, :, None].repeat(1, 1, N2, 1)
    y2 = B[:, None].repeat(1, N1, 1, 1)

    diff = torch.sum((y1 - y2) ** 2, dim=3)
    loss1, _ = torch.min(diff, dim=1)
    # loss2, _ = torch.min(diff, dim=2)
    A_ = A[:, :, None, None]
    loss2 = F.grid_sample(B_dtf, A_, mode='bilinear', padding_mode='border', align_corners=True)

    loss = torch.mean(loss1) + torch.mean(loss2)
    return loss