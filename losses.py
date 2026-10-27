
import torch
import torch.nn.functional as F


def dice_loss(prob, target, eps=1.0):
    dims = (1, 2, 3)
    inter = (prob * target).sum(dims)
    return 1 - ((2 * inter + eps) / (prob.sum(dims) + target.sum(dims) + eps)).mean()


def _erode(x):
    return torch.min(-F.max_pool2d(-x, (3, 1), 1, (1, 0)), -F.max_pool2d(-x, (1, 3), 1, (0, 1)))

def _dilate(x):
    return F.max_pool2d(x, 3, 1, 1)

def soft_skeleton(x, iters=10):
    skel = F.relu(x - _dilate(_erode(x)))
    for _ in range(iters):
        x = _erode(x)
        delta = F.relu(x - _dilate(_erode(x)))
        skel = skel + F.relu(delta - skel * delta)
    return skel

def cl_dice_loss(prob, target, iters=10, eps=1.0):
    sp, st = soft_skeleton(prob, iters), soft_skeleton(target, iters)
    t_prec = ((sp * target).sum((1, 2, 3)) + eps) / (sp.sum((1, 2, 3)) + eps)
    t_sens = ((st * prob).sum((1, 2, 3)) + eps) / (st.sum((1, 2, 3)) + eps)
    return (1 - 2 * t_prec * t_sens / (t_prec + t_sens)).mean()


def seg_loss(logits, target, w_bce=1.0, w_dice=1.0, w_cldice=0.0):
    loss = w_bce * F.binary_cross_entropy_with_logits(logits, target)
    prob = torch.sigmoid(logits.float())
    if w_dice:
        loss = loss + w_dice * dice_loss(prob, target)
    if w_cldice:
        loss = loss + w_cldice * cl_dice_loss(prob, target)
    return loss


def distill_loss(student_logits, teacher_logits, T=2.0):
    soft_t = torch.sigmoid(teacher_logits / T)
    return F.binary_cross_entropy_with_logits(student_logits / T, soft_t) * (T * T)
