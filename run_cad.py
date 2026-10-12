"""
run_cad.py
Run Candidate-Adaptive Discriminative MaxSim (CAD-MaxSim) ablations on top of base candidates.
Requires:
  outputs/base_candidates.pt
  outputs/query_embeddings.pt
  outputs/idf.pt
Produces:
  outputs/rankings_{original,idf,adaptive,cad}.json
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
    parser.add_argument('--topk', type=int, default=100)
    args = parser.parse_args()

    out = Path(args.output_dir)

    if not (out / 'idf.pt').exists():
        run(f'python -m cad.idf --data_root {args.data_root} --query_emb {out / "query_embeddings.pt"} '
            f'--output {out / "idf.pt"}')

    run(f'python -m cad.rerank --candidates {out / "base_candidates.pt"} '
        f'--query_emb {out / "query_embeddings.pt"} --idf {out / "idf.pt"} '
        f'--output_dir {args.output_dir} --topk {args.topk}')

    print('\nCAD pipeline complete.')


if __name__ == '__main__':
    main()
