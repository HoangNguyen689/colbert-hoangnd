"""
cad/weighting.py
Compute CAD weights from corpus IDF and candidate-set StdDev.
w_i = IDF_i^alpha * (StdDev_i + eps)^beta
Then normalize so mean weight = 1.
"""
import torch


def compute_weights(idf, std, alpha=0.0, beta=0.0, eps=1e-6):
    """
    idf: (Q_len,) tensor
    std: (Q_len,) tensor
    """
    idf = torch.clamp(idf, min=0.0)
    std = torch.clamp(std, min=0.0)
    weights = (idf ** alpha) * ((std + eps) ** beta)
    weights = weights / (weights.mean() + eps)
    return weights
