"""
evaluate.py
Compute nDCG@10, MRR@10, Recall@100 from rankings and qrels.
"""
import argparse
import json
from pathlib import Path

import torch
import numpy as np


def dcg(scores, k):
    scores = np.asarray(scores)[:k]
    if len(scores) == 0:
        return 0.0
    return np.sum((2 ** scores - 1) / np.log2(np.arange(2, len(scores) + 2)))


def ndcg_at_k(rankings, qrels, k=10):
    ndcgs = []
    for qid, ranked_pids in rankings.items():
        if qid not in qrels:
            continue
        rel = qrels[qid]
        gains = [rel.get(pid, 0) for pid in ranked_pids]
        ideal = sorted(rel.values(), reverse=True)
        ideal_dcg = dcg(ideal, k)
        actual_dcg = dcg(gains, k)
        ndcgs.append(actual_dcg / ideal_dcg if ideal_dcg > 0 else 0.0)
    return float(np.mean(ndcgs))


def mrr_at_k(rankings, qrels, k=10):
    rr = []
    for qid, ranked_pids in rankings.items():
        if qid not in qrels:
            continue
        rel = qrels[qid]
        for i, pid in enumerate(ranked_pids[:k]):
            if rel.get(pid, 0) > 0:
                rr.append(1.0 / (i + 1))
                break
        else:
            rr.append(0.0)
    return float(np.mean(rr))


def recall_at_k(rankings, qrels, k=100):
    recalls = []
    for qid, ranked_pids in rankings.items():
        if qid not in qrels:
            continue
        rel = set([pid for pid, score in qrels[qid].items() if score > 0])
        if len(rel) == 0:
            continue
        retrieved = set(ranked_pids[:k])
        recalls.append(len(rel & retrieved) / len(rel))
    return float(np.mean(recalls))


def evaluate(rankings, qrels):
    return {
        'nDCG@10': ndcg_at_k(rankings, qrels, k=10),
        'MRR@10': mrr_at_k(rankings, qrels, k=10),
        'Recall@100': recall_at_k(rankings, qrels, k=100),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--rankings', required=True)
    parser.add_argument('--qrels', default='outputs/qrels_test.json')
    args = parser.parse_args()

    if args.rankings.endswith('.pt'):
        results = torch.load(args.rankings, map_location='cpu')
        rankings = {qid: data['pids'] for qid, data in results.items()}
    else:
        with open(args.rankings, 'r', encoding='utf-8') as f:
            rankings = json.load(f)
    with open(args.qrels, 'r', encoding='utf-8') as f:
        qrels = json.load(f)

    metrics = evaluate(rankings, qrels)
    for k, v in metrics.items():
        print(f'{k}: {v:.4f}')


if __name__ == '__main__':
    main()
