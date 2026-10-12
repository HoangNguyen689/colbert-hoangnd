"""
base/compress_residuals.py
Compute residuals from centroids/codes and quantize them with 2-bit per dimension.
Outputs:
  outputs/residuals.pt -> {cutoffs, bucket_weights, idx or packed}
"""
import argparse
from pathlib import Path

import torch

from base.residual import compute_residuals, fit_residual_buckets, quantize_residuals, dequantize_residuals, pack_bits


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--emb_path', default='outputs/doc_embeddings.pt')
    parser.add_argument('--centroids_path', default='outputs/centroids.pt')
    parser.add_argument('--codes_path', default='outputs/codes.pt')
    parser.add_argument('--output_dir', default='outputs')
    parser.add_argument('--nbits', type=int, default=2)
    parser.add_argument('--no-pack', dest='pack', action='store_false',
                        help='keep residuals as uint8 tensor instead of bit-packing')
    parser.set_defaults(pack=True)
    args = parser.parse_args()

    D = torch.load(args.emb_path, map_location='cpu')['D']
    centroids = torch.load(args.centroids_path, map_location='cpu')
    codes = torch.load(args.codes_path, map_location='cpu')

    print('Computing residuals...')
    residuals = compute_residuals(D, centroids, codes)

    print('Fitting buckets...')
    cutoffs, bucket_weights = fit_residual_buckets(residuals, nbits=args.nbits)

    print('Quantizing residuals...')
    idx = quantize_residuals(residuals, cutoffs)

    recon = dequantize_residuals(idx, bucket_weights)
    mse = (residuals - recon).pow(2).mean().item()
    print(f'Residual reconstruction MSE: {mse:.6f}')

    out = {'cutoffs': cutoffs, 'bucket_weights': bucket_weights}
    if args.pack:
        print('Packing bits...')
        out['packed'] = torch.from_numpy(pack_bits(idx, nbits=args.nbits))
    else:
        out['idx'] = idx

    out_path = Path(args.output_dir) / 'residuals.pt'
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, out_path)
    print(f'Saved residuals -> {out_path}')


if __name__ == '__main__':
    main()
