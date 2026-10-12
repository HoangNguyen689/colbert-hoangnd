"""
compare_official.py
Compare Mini-ColBERTv2 metrics against official ColBERTv2 baseline on SciFact.
Official baseline (checkpoint colbert-ir/colbertv2.0):
  nDCG@10  = 0.6438
  MRR@10   = 0.6120
  Recall@100 = 0.8842
"""
import argparse
import json
from pathlib import Path

from mini_colbert.evaluate import evaluate


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
    parser.add_argument('--rankings', required=True, help='Mini-ColBERTv2 rankings JSON')
    parser.add_argument('--qrels', default='outputs/qrels_test.json')
    parser.add_argument('--official_rankings', default=None, help='Optional official rankings JSON')
    args = parser.parse_args()

    with open(args.rankings, 'r', encoding='utf-8') as f:
        mini_rankings = json.load(f)
    with open(args.qrels, 'r', encoding='utf-8') as f:
        qrels = json.load(f)

    mini_metrics = evaluate(mini_rankings, qrels)

    official = OFFICIAL_BASELINE
    if args.official_rankings:
        with open(args.official_rankings, 'r', encoding='utf-8') as f:
            official_rankings = json.load(f)
        official = evaluate(official_rankings, qrels)

    compare(mini_metrics, official)


if __name__ == '__main__':
    main()
