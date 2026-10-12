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
    Fit uniform quantization buckets per dimension.
    Returns (lowers, steps, levels) where levels = 2**nbits.
    For official-like quality one could fit k-means per dim; uniform is simple & fast.
    """
    levels = 2 ** nbits
    lo = residuals.min(dim=0).values
    hi = residuals.max(dim=0).values
    step = (hi - lo) / levels
    return lo, step, levels


def quantize_residuals(residuals, lo, step, levels):
    """
    Quantize residuals to bucket indices in [0, levels).
    Returns (N, dim) uint8 indices.
    """
    norm = (residuals - lo) / (step + 1e-8)
    idx = torch.clamp(norm.long(), 0, levels - 1)
    return idx


def dequantize_residuals(idx, lo, step):
    """
    idx: (N, dim) long in [0, levels)
    returns residuals_hat (N, dim)
    """
    return lo + (idx + 0.5) * step


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
