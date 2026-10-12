"""
base/residual.py
2-bit residual quantization / dequantization.
"""
import torch
import numpy as np


def compute_residuals(D, centroids, codes):
    return D - centroids[codes]


def fit_residual_buckets(residuals, nbits=2):
    levels = 2 ** nbits
    quantiles = torch.linspace(1.0 / levels, 1.0 - 1.0 / levels, levels - 1)
    cutoffs = torch.quantile(residuals, quantiles, dim=0)

    idx = quantize_residuals(residuals, cutoffs)
    bucket_weights = torch.zeros(levels, residuals.shape[1], dtype=residuals.dtype)
    for b in range(levels):
        mask = (idx == b).float()
        bucket_weights[b] = (residuals * mask).sum(0) / (mask.sum(0) + 1e-8)
    return cutoffs, bucket_weights


def quantize_residuals(residuals, cutoffs):
    residuals = residuals.contiguous()
    idx = torch.zeros_like(residuals, dtype=torch.uint8)
    for d in range(residuals.shape[1]):
        idx[:, d] = torch.bucketize(residuals[:, d].contiguous(), cutoffs[:, d].contiguous())
    return idx


def dequantize_residuals(idx, bucket_weights):
    N, dim = idx.shape
    idx = idx.long()
    dims = torch.arange(dim, device=idx.device).unsqueeze(0).expand(N, dim)
    return bucket_weights[idx, dims]


def pack_bits(idx, nbits=2):
    """Vectorized bit-packing. Returns a 1-D uint8 numpy array.

    Supports any nbits that divides 8 (1, 2, 4, 8). For nbits=2 and dim=128,
    each token consumes exactly 32 bytes.
    """
    idx = idx.cpu().numpy().astype(np.uint8)
    N, dim = idx.shape
    vals_per_byte = 8 // nbits
    mask = (1 << nbits) - 1
    pad = (vals_per_byte - dim % vals_per_byte) % vals_per_byte
    if pad:
        idx = np.pad(idx, ((0, 0), (0, pad)), constant_values=0)

    flat = idx.reshape(-1, vals_per_byte)
    shifts = (np.arange(vals_per_byte, dtype=np.uint8) * nbits)
    packed = np.bitwise_or.reduce((flat & mask) << shifts, axis=1).astype(np.uint8)
    return packed


def unpack_bits(packed, shape, nbits=2):
    """Vectorized inverse of pack_bits."""
    N, dim = shape
    vals_per_byte = 8 // nbits
    mask = (1 << nbits) - 1
    dim_padded = (dim + vals_per_byte - 1) // vals_per_byte * vals_per_byte
    n_bytes_expected = N * dim_padded // vals_per_byte

    packed = np.asarray(packed, dtype=np.uint8).reshape(-1)
    if packed.shape[0] < n_bytes_expected:
        packed = np.pad(packed, (0, n_bytes_expected - packed.shape[0]), constant_values=0)
    elif packed.shape[0] > n_bytes_expected:
        packed = packed[:n_bytes_expected]

    flat = packed.reshape(-1, 1)
    shifts = (np.arange(vals_per_byte, dtype=np.uint8) * nbits)
    idx = ((flat >> shifts[None, :]) & mask).reshape(N, dim_padded)
    if pad := dim_padded - dim:
        idx = idx[:, :-pad]
    return torch.from_numpy(idx)
