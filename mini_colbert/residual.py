"""
residual.py
2-bit residual quantization / dequantization.
For each token embedding D_i:
  centroid = centroids[code_i]
  residual = D_i - centroid
  residual is quantized per-dimension to 2 bits (4 bins).
"""
import torch
import numpy as np


def compute_residuals(D, centroids, codes):
    """
    D: (N, dim) float
    centroids: (K, dim) float
    codes: (N,) long
    returns residuals (N, dim)
    """
    return D - centroids[codes]


def fit_residual_buckets(residuals, nbits=2):
    """
    Fit official-style quantile buckets per dimension.
    Returns (cutoffs, bucket_weights) where levels = 2**nbits.
    cutoffs: (levels-1, dim)
    bucket_weights: (levels, dim) mean residual value in each bucket
    """
    levels = 2 ** nbits
    # Quantile cutoffs divide residuals into equal-frequency buckets
    quantiles = torch.linspace(1.0 / levels, 1.0 - 1.0 / levels, levels - 1)
    cutoffs = torch.quantile(residuals, quantiles, dim=0)  # (levels-1, dim)

    # Reconstruction weights: mean residual in each bucket per dimension
    idx = quantize_residuals(residuals, cutoffs)  # (N, dim) in [0, levels)
    bucket_weights = torch.zeros(levels, residuals.shape[1], dtype=residuals.dtype)
    for b in range(levels):
        mask = (idx == b).float()
        bucket_weights[b] = (residuals * mask).sum(0) / (mask.sum(0) + 1e-8)
    return cutoffs, bucket_weights


def quantize_residuals(residuals, cutoffs):
    """
    Quantize residuals to bucket indices in [0, levels).
    cutoffs: (levels-1, dim)
    Returns (N, dim) uint8 indices.
    """
    residuals = residuals.contiguous()
    idx = torch.zeros_like(residuals, dtype=torch.uint8)
    for d in range(residuals.shape[1]):
        idx[:, d] = torch.bucketize(residuals[:, d].contiguous(), cutoffs[:, d].contiguous())
    return idx


def dequantize_residuals(idx, bucket_weights):
    """
    idx: (N, dim) long/uint8 in [0, levels)
    bucket_weights: (levels, dim)
    returns residuals_hat (N, dim)
    """
    N, dim = idx.shape
    idx = idx.long()
    dims = torch.arange(dim, device=idx.device).unsqueeze(0).expand(N, dim)
    return bucket_weights[idx, dims]


def pack_bits(idx, nbits=2):
    """
    Pack uint8 indices into compact bytes.
    idx: (N, dim) tensor with values < 2**nbits.
    Returns numpy array of packed bytes and metadata for unpacking.
    """
    N, dim = idx.shape
    bits = idx.cpu().numpy().astype(np.uint8)
    total_bits = N * dim * nbits
    n_bytes = (total_bits + 7) // 8
    packed = np.zeros(n_bytes, dtype=np.uint8)

    bit_pos = 0
    for i in range(N):
        for j in range(dim):
            val = bits[i, j]
            byte_idx = bit_pos // 8
            offset = bit_pos % 8
            packed[byte_idx] |= (val << offset) & 0xFF
            if offset + nbits > 8:
                packed[byte_idx + 1] |= (val >> (8 - offset))
            bit_pos += nbits

    return packed


def unpack_bits(packed, shape, nbits=2):
    """
    Unpack compact bytes back to uint8 indices.
    packed: numpy bytes
    shape: (N, dim)
    """
    N, dim = shape
    total_bits = N * dim * nbits
    idx = np.zeros((N, dim), dtype=np.uint8)

    bit_pos = 0
    mask = (1 << nbits) - 1
    for i in range(N):
        for j in range(dim):
            byte_idx = bit_pos // 8
            offset = bit_pos % 8
            val = (packed[byte_idx] >> offset) & mask
            if offset + nbits > 8:
                val |= (packed[byte_idx + 1] << (8 - offset)) & mask
            idx[i, j] = val
            bit_pos += nbits

    return torch.from_numpy(idx)
