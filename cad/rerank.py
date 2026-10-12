"""
cad/rerank.py
Candidate-Adaptive Discriminative MaxSim (CAD-MaxSim) reranking from base candidate artifact.
"""
import argparse
import json
from pathlib import Path

import torch

from cad.weighting import compute_weights


def rerank_query(token_scores, idf_vals, alpha=0.0, beta=0.0):
    """
    token_scores: (N_candidates, Q_len)
    idf_vals: (Q_len,)
    Returns weighted scores (N_candidates,)
    """
    std_vals = token_scores.std(dim=0)
    weights = compute_weights(idf_vals, std_vals, alpha=alpha, beta=beta)
    return (token_scores * weights.unsqueeze(0)).sum(dim=1)


def rerank_all(base_candidates, idf_dict, query_token_ids, qids, alpha=0.0, beta=0.0):
    rankings = {}
    for qid in qids:
        if qid not in base_candidates:
            rankings[qid] = []
            continue
        cand = base_candidates[qid]
        token_ids = query_token_ids[qids.index(qid)].tolist()
        idf_vals = torch.tensor([idf_dict.get(tid, 1.0) for tid in token_ids], dtype=torch.float32)

        scores = rerank_query(cand['token_scores'], idf_vals, alpha=alpha, beta=beta)
        sorted_idx = scores.argsort(descending=True)
        rankings[qid] = [cand['pids'][i] for i in sorted_idx.tolist()]
    return rankings


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--candidates', default='outputs/base_candidates.pt')
    parser.add_argument('--query_emb', default='outputs/query_embeddings.pt')
    parser.add_argument('--idf', default='outputs/idf.pt')
    parser.add_argument('--output_dir', default='outputs')
    parser.add_argument('--topk', type=int, default=100)
    args = parser.parse_args()

    base_candidates = torch.load(args.candidates, map_location='cpu')
    qdata = torch.load(args.query_emb, map_location='cpu')
    idf_dict = torch.load(args.idf, map_location='cpu')

    qids = qdata['qids']
    token_ids = qdata['token_ids']

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    variants = {
        'original': (0.0, 0.0),
        'idf': (1.0, 0.0),
        'adaptive': (0.0, 1.0),
        'cad': (1.0, 1.0),
    }

    for name, (alpha, beta) in variants.items():
        print(f'Running {name} (alpha={alpha}, beta={beta})...')
        rankings = rerank_all(base_candidates, idf_dict, token_ids, qids, alpha=alpha, beta=beta)
        out_path = out_dir / f'rankings_{name}.json'
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump({qid: pids[:args.topk] for qid, pids in rankings.items()}, f)
        print(f'  Saved -> {out_path}')


if __name__ == '__main__':
    main()
