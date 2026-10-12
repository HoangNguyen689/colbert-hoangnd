"""
evaluation/compare_official.py
Compare Mini-ColBERTv2 metrics against official ColBERTv2 baseline on SciFact.
"""
import argparse
from pathlib import Path

from evaluation.metrics import evaluate, load_json


OFFICIAL_BASELINE = {
    'nDCG@10': 0.6438,
    'MRR@10': 0.6120,
    'Recall@100': 0.8842,
}


def compare(metrics, official=OFFICIAL_BASELINE):
    print(f"{'Metric':<12} {'Mini':>8} {'Official':>10} {'Diff':>8}")
    print('-' * 42)
    for k in ['nDCG@10', 'MRR@10', 'Recall@100']:
        mini = metrics.get(k, 0.0)
        off = official.get(k, 0.0)
        diff = mini - off
        print(f"{k:<12} {mini:>8.4f} {off:>10.4f} {diff:>+8.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--rankings', required=True)
    parser.add_argument('--qrels', default='outputs/qrels_test.json')
    parser.add_argument('--official_rankings', default=None)
    args = parser.parse_args()

    mini_rankings = load_json(args.rankings)
    qrels = load_json(args.qrels)

    mini_metrics = evaluate(mini_rankings, qrels)

    official = OFFICIAL_BASELINE
    if args.official_rankings:
        official_rankings = load_json(args.official_rankings)
        official = evaluate(official_rankings, qrels)

    compare(mini_metrics, official)


if __name__ == '__main__':
    main()
