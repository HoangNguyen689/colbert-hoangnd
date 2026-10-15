"""
base/residual.py
2-bit residual quantization / dequantization.
"""
import torch
import numpy as np


def compute_residuals(D, centroids, codes, chunk=None):
    """D - centroids[codes].

    With chunk=None this is a single (N, dim) subtraction. Pass a chunk size to
    avoid materialising a second full (N, dim) gather, which matters when the
    embedding tensor is several GB (e.g. fiqa on a 12.7 GB Colab runtime).
    """
    if chunk is None:
        return D - centroids[codes]
    # Match the dtype of the unchunked expression: D may be stored in half
    # precision while centroids are float32, and D - centroids[codes] promotes to
    # float32. torch.empty_like(D) would silently downcast that back to half.
    out = torch.empty(D.shape, dtype=torch.result_type(D, centroids))
    for start in range(0, D.shape[0], chunk):
        end = start + chunk
        out[start:end] = D[start:end] - centroids[codes[start:end]]
    return out


def fit_residual_buckets(residuals, nbits=2, chunk=262144):
    """Fit per-dimension quantile cutoffs and per-bucket mean weights.

    Memory-lean on purpose: cutoffs are computed column by column (torch.quantile
    sorts its input, so passing the whole (N, dim) tensor needs an extra full
    copy) and bucket weights are accumulated chunk by chunk instead of building a
    (N, dim) mask per bucket.
    """
    levels = 2 ** nbits
    dim = residuals.shape[1]
    N = residuals.shape[0]
    quantiles = torch.linspace(1.0 / levels, 1.0 - 1.0 / levels, levels - 1)

    cutoffs = torch.empty(levels - 1, dim, dtype=residuals.dtype)
    for d in range(dim):
        cutoffs[:, d] = torch.quantile(residuals[:, d], quantiles)

    weights = torch.zeros(levels, dim, dtype=torch.float32)
    counts = torch.zeros(levels, dim, dtype=torch.float32)
    for start in range(0, N, chunk):
        block = residuals[start:start + chunk]
        idx = quantize_residuals(block, cutoffs)
        for b in range(levels):
            m = (idx == b)
            weights[b] += (block * m).sum(0)
            counts[b] += m.sum(0).to(torch.float32)
    bucket_weights = (weights / (counts + 1e-8)).to(residuals.dtype)
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
