"""
baselines/official/evaluate_official.py
Compute nDCG@10, MRR@10, Recall@100 from an official ColBERT ranking file,
and optionally write the metrics to JSON (consumed by run_evaluate.py).

Standalone on purpose: copy into the official ColBERT repo working directory.
"""
from pathlib import Path
from collections import defaultdict
import argparse
import csv
import json
import math


def load_qrels(path):
    qrels = defaultdict(dict)
    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            qrels[int(row["qid"])][int(row["pid"])] = int(row["rel"])
    return qrels


def load_ranking(path):
    ranking = defaultdict(list)
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip().split("\t")
            if len(parts) < 3:
                continue
            qid, pid = int(parts[0]), int(parts[1])
            rank = int(parts[2])
            score = float(parts[3]) if len(parts) > 3 else 0.0
            ranking[qid].append((rank, pid, score))
    for qid in ranking:
        ranking[qid].sort()
    return ranking


def dcg(rels):
    return sum((2 ** rel - 1) / math.log2(i + 2) for i, rel in enumerate(rels))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="scifact")
    ap.add_argument("--qrels", default=None, help="default: data/<dataset>_colbert/qrels.tsv")
    ap.add_argument("--ranking", default=None, help="default: ranking.<dataset>.tsv")
    ap.add_argument("--json-out", default=None,
                    help="write {nDCG@10, MRR@10, Recall@100} here")
    args = ap.parse_args()

    ds = args.dataset
    qrels_path = args.qrels or f"data/{ds}_colbert/qrels.tsv"
    ranking_path = args.ranking or f"ranking.{ds}.tsv"

    qrels = load_qrels(qrels_path)
    ranking = load_ranking(ranking_path)

    ndcg10 = []
    mrr10 = []
    recall100 = []

    for qid, rels in qrels.items():
        retrieved = [pid for _, pid, _ in ranking.get(qid, [])]
        gains10 = [rels.get(pid, 0) for pid in retrieved[:10]]
        ideal10 = sorted(rels.values(), reverse=True)[:10]
        denom = dcg(ideal10)
        ndcg10.append(dcg(gains10) / denom if denom else 0.0)

        rr = 0.0
        for i, pid in enumerate(retrieved[:10], 1):
            if rels.get(pid, 0) > 0:
                rr = 1.0 / i
                break
        mrr10.append(rr)

        relevant = {pid for pid, rel in rels.items() if rel > 0}
        hit = len(relevant.intersection(retrieved[:100]))
        recall100.append(hit / len(relevant) if relevant else 0.0)

    mean = lambda xs: sum(xs) / len(xs) if xs else 0.0
    metrics = {
        "nDCG@10": round(mean(ndcg10), 4),
        "MRR@10": round(mean(mrr10), 4),
        "Recall@100": round(mean(recall100), 4),
    }
    print(f"dataset={ds}")
    print(f"queries={len(qrels)}")
    for k, v in metrics.items():
        print(f"{k}={v:.4f}")

    if args.json_out:
        Path(args.json_out).write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        print("JSON ->", args.json_out)


if __name__ == "__main__":
    main()
