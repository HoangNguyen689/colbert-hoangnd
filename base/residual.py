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
