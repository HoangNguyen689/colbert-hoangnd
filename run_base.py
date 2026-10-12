"""
run_base.py
End-to-end Mini-ColBERTv2 BASE pipeline.
Produces:
  outputs/base_candidates.pt
  outputs/rankings_original.json
"""
import argparse
import subprocess
import sys
from pathlib import Path


def run(cmd):
    print(f'\n>>> {cmd}')
    ret = subprocess.call(cmd, shell=True)
    if ret != 0:
        sys.exit(ret)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', default='data')
    parser.add_argument('--output_dir', default='outputs')
    parser.add_argument('--K', type=int, default=8192)
    parser.add_argument('--n_samples', type=int, default=200000)
    parser.add_argument('--ncells', type=int, default=4)
    parser.add_argument('--topk', type=int, default=100)
    parser.add_argument('--device', default=None)
    parser.add_argument('--doc_batch_size', type=int, default=32)
    parser.add_argument('--query_batch_size', type=int, default=32)
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    device = args.device or 'cpu'

    run(f'python -m base.encode --data_root {args.data_root} --output_dir {args.output_dir} '
        f'--doc_batch_size {args.doc_batch_size} --query_batch_size {args.query_batch_size}')
    run(f'python -m base.index --emb_path {out / "doc_embeddings.pt"} --output_dir {args.output_dir} '
        f'--K {args.K} --n_samples {args.n_samples}')
    run(f'python -m base.compress_residuals --emb_path {out / "doc_embeddings.pt"} '
        f'--centroids_path {out / "centroids.pt"} --codes_path {out / "codes.pt"} '
        f'--output_dir {args.output_dir} --nbits 2')
    run(f'python -m base.ivf --metadata_path {out / "metadata.pt"} --codes_path {out / "codes.pt"} '
        f'--output_dir {args.output_dir}')
    run(f'python -m base.search --index_dir {args.output_dir} --query_emb {out / "query_embeddings.pt"} '
        f'--output_candidates {out / "base_candidates.pt"} --output_ranking {out / "rankings_original.json"} '
        f'--ncells {args.ncells} --topk {args.topk} --device {device}')

    print('\nBase pipeline complete.')


if __name__ == '__main__':
    main()
