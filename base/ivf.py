"""
base/ivf.py
Build an inverted file index: centroid_id -> list of passage ids.
Outputs:
  outputs/ivf.pt -> {centroid_id: [pid_index, ...]}
"""
import argparse
from pathlib import Path
from collections import defaultdict

import torch


def build_ivf(codes, doclens):
    ivf = defaultdict(list)
    offset = 0
    for pid_idx, length in enumerate(doclens):
        unique_centroids = torch.unique(codes[offset:offset+length]).tolist()
        for cid in unique_centroids:
            ivf[cid].append(pid_idx)
        offset += length
    return dict(ivf)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--metadata_path', default='outputs/metadata.pt')
    parser.add_argument('--codes_path', default='outputs/codes.pt')
    parser.add_argument('--output_dir', default='outputs')
    args = parser.parse_args()

    metadata = torch.load(args.metadata_path, map_location='cpu')
    codes = torch.load(args.codes_path, map_location='cpu')
    doclens = metadata['doclens']

    ivf = build_ivf(codes, doclens)
    out_path = Path(args.output_dir) / 'ivf.pt'
    torch.save(ivf, out_path)
    print(f'Saved IVF with {len(ivf)} centroids -> {out_path}')


if __name__ == '__main__':
    main()
