"""
run_evaluate.py
Evaluate all ranking files and print a comparison table.

The "Official ColBERTv2" row is only shown when an official baseline is known
for the dataset:
  1. outputs/<dataset>/official_baseline.json  (produced by baselines/official), or
  2. the built-in OFFICIAL_BASELINES table below.

Baselines are measured with the same settings as the mini pipeline
(query_maxlen=32, doc_maxlen=180, nbits=2, top-100), so the comparison is fair.
"""
import argparse
import json
from pathlib import Path

from evaluation.metrics import evaluate, load_json


# Same-settings (query_maxlen=32, doc_maxlen=180, nbits=2) official ColBERTv2
# numbers measured with baselines/official on each dataset.
OFFICIAL_BASELINES = {
    'scifact': {'nDCG@10': 0.6438, 'MRR@10': 0.6120, 'Recall@100': 0.8842},
}


def load_official_baseline(dataset, out_dir):
    """Prefer outputs/<ds>/official_baseline.json; otherwise the built-in table."""
    path = Path(out_dir) / 'official_baseline.json'
    if path.exists():
        metrics = json.loads(path.read_text(encoding='utf-8'))
        # keep only the metric keys we know how to print
        return {k: float(metrics[k]) for k in ('nDCG@10', 'MRR@10', 'Recall@100') if k in metrics}
    return OFFICIAL_BASELINES.get(dataset)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', default='outputs')
    parser.add_argument('--qrels', default='outputs/qrels_test.json')
    parser.add_argument('--dataset', default='scifact',
                        help='selects which official baseline to display, if any')
    args = parser.parse_args()

    qrels = load_json(args.qrels)
    out = Path(args.output_dir)

    variants = []
    official = load_official_baseline(args.dataset, args.output_dir)
    if official:
        variants.append(('Official ColBERTv2', None, official))
    variants += [
        ('Mini Base', out / 'rankings_original.json', None),
        ('IDF', out / 'rankings_idf.json', None),
        ('Adaptive', out / 'rankings_adaptive.json', None),
        ('CAD', out / 'rankings_cad.json', None),
    ]

    print(f"Dataset: {args.dataset}")
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
