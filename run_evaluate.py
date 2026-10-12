"""
run_evaluate.py
Evaluate all ranking files and print comparison table.
"""
import argparse
from pathlib import Path

from evaluation.metrics import evaluate, load_json


OFFICIAL_BASELINE = {
    'nDCG@10': 0.6438,
    'MRR@10': 0.6120,
    'Recall@100': 0.8842,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', default='outputs')
    parser.add_argument('--qrels', default='outputs/qrels_test.json')
    args = parser.parse_args()

    qrels = load_json(args.qrels)
    out = Path(args.output_dir)

    variants = [
        ('Official ColBERTv2', None, OFFICIAL_BASELINE),
        ('Mini Base', out / 'rankings_original.json', None),
        ('IDF', out / 'rankings_idf.json', None),
        ('Adaptive', out / 'rankings_adaptive.json', None),
        ('CAD', out / 'rankings_cad.json', None),
    ]

    print(f"{'Method':<22} {'nDCG@10':>10} {'MRR@10':>10} {'Recall@100':>12}")
    print('-' * 60)
    for name, path, baseline in variants:
        if baseline:
            metrics = baseline
        elif path and path.exists():
            rankings = load_json(path)
            metrics = evaluate(rankings, qrels)
        else:
            metrics = {'nDCG@10': 0.0, 'MRR@10': 0.0, 'Recall@100': 0.0}
        print(f"{name:<22} {metrics['nDCG@10']:>10.4f} {metrics['MRR@10']:>10.4f} {metrics['Recall@100']:>12.4f}")


if __name__ == '__main__':
    main()
