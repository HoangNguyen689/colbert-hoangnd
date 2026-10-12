"""
evaluation/metrics.py
Compute nDCG@10, MRR@10, Recall@100 from rankings and qrels.
"""
import json
import numpy as np


def dcg(scores, k):
    scores = np.asarray(scores)[:k]
    if len(scores) == 0:
        return 0.0
    return np.sum((2 ** scores - 1) / np.log2(np.arange(2, len(scores) + 2)))


def ndcg_at_k(rankings, qrels, k=10):
    ndcgs = []
    for qid, rel in qrels.items():
        ranked_pids = rankings.get(qid, [])
        gains = [rel.get(pid, 0) for pid in ranked_pids]
        ideal = sorted(rel.values(), reverse=True)
        ideal_dcg = dcg(ideal, k)
        actual_dcg = dcg(gains, k)
        ndcgs.append(actual_dcg / ideal_dcg if ideal_dcg > 0 else 0.0)
    return float(np.mean(ndcgs))


def mrr_at_k(rankings, qrels, k=10):
    rr = []
    for qid, rel in qrels.items():
        ranked_pids = rankings.get(qid, [])
        for i, pid in enumerate(ranked_pids[:k]):
            if rel.get(pid, 0) > 0:
                rr.append(1.0 / (i + 1))
                break
        else:
            rr.append(0.0)
    return float(np.mean(rr))


def recall_at_k(rankings, qrels, k=100):
    recalls = []
    for qid, rel in qrels.items():
        rel_set = set([pid for pid, score in rel.items() if score > 0])
        if len(rel_set) == 0:
            continue
        ranked_pids = rankings.get(qid, [])
        retrieved = set(ranked_pids[:k])
        recalls.append(len(rel_set & retrieved) / len(rel_set))
    return float(np.mean(recalls))


def evaluate(rankings, qrels):
    return {
        'nDCG@10': ndcg_at_k(rankings, qrels, k=10),
        'MRR@10': mrr_at_k(rankings, qrels, k=10),
        'Recall@100': recall_at_k(rankings, qrels, k=100),
    }


def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)
