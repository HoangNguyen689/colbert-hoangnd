"""
base/maxsim.py
Pure PyTorch MaxSim implementations.
"""
import torch


def maxsim_per_token(Q, D):
    """
    Q: (Q_len, dim)
    D: (doc_len, dim)
    Returns c_i(d) = max_j (Q_i · D_j) for each query token i.
    """
    sim = Q @ D.T  # (Q_len, doc_len)
    return sim.max(dim=-1).values  # (Q_len,)


def maxsim_score(Q, D):
    """
    Returns sum_i c_i(d).
    """
    return maxsim_per_token(Q, D).sum()
