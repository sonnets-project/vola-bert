# Code adapted directly from the reproduction of the GPT4TS paper done by liaoyuhua
# Original source: https://github.com/liaoyuhua/GPT-TS
# Credit to the authors of the paper and repository.

import torch
import torch.nn.functional as F


def mae_loss(prediction, target):
    """
    Arguments:
        prediction: shape (B, N, L)
        target    : shape (B, N, L)
    """
    mask = ~torch.isnan(target)
    masked_prediction = prediction[mask]
    masked_target = target[mask]
    return F.l1_loss(masked_prediction, masked_target)


def mse_loss(prediction, target):
    """
    Arguments:
        prediction: shape (B, N, L)
        target    : shape (B, N, L)
    """
    mask = ~torch.isnan(target)
    masked_prediction = prediction[mask]
    masked_target = target[mask]
    return F.mse_loss(masked_prediction, masked_target)
