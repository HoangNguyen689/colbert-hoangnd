"""
cad/signals.py
Per-query-token "importance" signals for CAD-MaxSim.

Every signal takes the (n_candidates, q_len) tensor of per-token MaxSim values
(the `token_scores` stored inside base_candidates.pt) and returns a (q_len,)
tensor of non-negative weights.

`std` is the signal used by the original CAD-MaxSim. The others are candidate
replacements: the hypothesis is that dispersion (std) measures spread, not
discriminativeness, and is too noisy on a 200-passage candidate pool.
"""
import torch

EPS = 1e-6


def sig_std(ts):
    """Standard deviation over the candidate pool (original CAD signal)."""
    return ts.std(dim=0)


def sig_mad(ts):
    """Median absolute deviation: same idea as std but robust to outliers."""
    med = ts.median(dim=0).values
    return (ts - med).abs().median(dim=0).values


def sig_gap(ts):
    """Top-1 minus mean: how far the best passage stands out for this token."""
    return ts.max(dim=0).values - ts.mean(dim=0)


def sig_margin(ts):
    """Top-1 minus top-2: how steeply this token separates the best passage."""
    if ts.shape[0] < 2:
        return torch.zeros(ts.shape[1], dtype=ts.dtype, device=ts.device)
    top = ts.topk(2, dim=0).values
    return top[0] - top[1]


def sig_cv(ts):
    """Coefficient of variation: spread normalised by the mean level."""
    return ts.std(dim=0) / (ts.mean(dim=0).abs() + EPS)


def sig_snr(ts):
    """(max - mean) / std: signal-to-noise of this token over the pool."""
    return (ts.max(dim=0).values - ts.mean(dim=0)) / (ts.std(dim=0) + EPS)


def sig_ldf(ts):
    """Local document frequency: fraction of candidates this token matches
    positively. A candidate-set analogue of corpus IDF."""
    return (ts > 0).float().mean(dim=0)


SIGNALS = {
    'std': sig_std,
    'mad': sig_mad,
    'gap': sig_gap,
    'margin': sig_margin,
    'cv': sig_cv,
    'snr': sig_snr,
    'ldf': sig_ldf,
}
