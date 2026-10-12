"""
cad_maxsim.py
Context-Aware Diversified MaxSim (CAD-MaxSim) weighting.
Given per-query-token MaxSim scores c_i(d) over a candidate set, compute:
  IDF_i    = inverse document frequency of token i in candidate set
  StdDev_i = standard deviation of c_i(d) over candidate set
Then score(d) = sum_i (1 + alpha * IDF_i + beta * StdDev_i) * c_i(d)
Ablations:
  Original  (alpha=0, beta=0)
  IDF       (alpha=1, beta=0)
  Adaptive  (alpha=0, beta=1)
  CAD       (alpha=1, beta=1)
"""
import argparse
import json
from pathlib import Path

import torch
import numpy as np

from mini_colbert.search import ColBERTSearcher


def compute_idf(token_scores, candidate_mask):
    """
    token_scores: (Q_len, n_candidates)
    candidate_mask: (n_candidates,) bool
    Returns IDF per query token: log(N / df)
    """
    present = (token_scores[:, candidate_mask] > 0).float().sum(dim=1)  # df
    N = candidate_mask.sum().item()
    idf = torch.log((N + 1) / (present + 1))
    return idf


def compute_std(token_scores, candidate_mask):
    """
    token_scores: (Q_len, n_candidates)
    Returns std dev per query token over candidates.
    """
    masked = token_scores[:, candidate_mask]
    return masked.std(dim=1)


def cad_maxsim(token_scores, alpha=0.0, beta=0.0, normalize_weights=True):
    """
    token_scores: (Q_len, n_candidates) tensor
    Returns weighted scores (n_candidates,) and weights (Q_len,).
    """
    n_candidates = token_scores.shape[1]
    candidate_mask = torch.ones(n_candidates, dtype=torch.bool)

    idf = compute_idf(token_scores, candidate_mask)
    std = compute_std(token_scores, candidate_mask)

    if normalize_weights:
        idf = (idf - idf.mean()) / (idf.std() + 1e-8)
        std = (std - std.mean()) / (std.std() + 1e-8)

    weights = 1.0 + alpha * idf + beta * std
    weighted_scores = (weights.unsqueeze(1) * token_scores).sum(dim=0)
    return weighted_scores, weights


def rerank_results(searcher, qdata, qrels, alpha=0.0, beta=0.0, ncells=4, topk=100):
    """
    Re-rank candidate set from approximate search using CAD-MaxSim.
    Returns rankings dict: qid -> list of pids.
    """
    device = searcher.centroids.device
    Qs = qdata['Q'].to(device)
    masks = qdata.get('masks', None)
    if masks is not None:
        masks = masks.to(device).bool()
    qids = qdata['qids']

    rankings = {}
    for qid, Q in zip(qids, Qs):
        mask = None
        if masks is not None:
            mask = masks[len(rankings)]
        Q = torch.nn.functional.normalize(Q, p=2, dim=-1)
        candidates = searcher.get_candidates(Q, ncells=ncells)
        if len(candidates) == 0:
            rankings[qid] = []
            continue

        emb, doclens_cand = searcher.reconstruct_candidate_embeddings(candidates)
        sim = Q @ emb.T  # (Q_len, total_tokens)
        if mask is not None:
            sim = sim[mask]

        token_scores = []
        offset = 0
        for length in doclens_cand:
            doc_sim = sim[:, offset:offset+length]
            token_scores.append(doc_sim.max(dim=-1).values)
            offset += length
        token_scores = torch.stack(token_scores, dim=1)  # (Q_len, n_candidates)

        weighted_scores, weights = cad_maxsim(token_scores, alpha=alpha, beta=beta)
        sorted_idx = weighted_scores.argsort(descending=True)
        top_pids = [searcher.pids[candidates[i]] for i in sorted_idx[:topk].tolist()]
        rankings[qid] = top_pids

    return rankings


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--index_dir', default='outputs')
    parser.add_argument('--query_emb', default='outputs/query_embeddings.pt')
    parser.add_argument('--qrels', default='outputs/qrels_test.json')
    parser.add_argument('--output_dir', default='outputs')
    parser.add_argument('--ncells', type=int, default=4)
    parser.add_argument('--topk', type=int, default=100)
    parser.add_argument('--device', default=None)
    args = parser.parse_args()

    device = args.device or ('mps' if torch.backends.mps.is_available() else 'cpu')
    searcher = ColBERTSearcher(args.index_dir, device=device)
    qdata = torch.load(args.query_emb, map_location='cpu')

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ablations = {
        'original': (0.0, 0.0),
        'idf': (1.0, 0.0),
        'adaptive': (0.0, 1.0),
        'cad': (1.0, 1.0),
    }

    for name, (alpha, beta) in ablations.items():
        print(f'Running {name} (alpha={alpha}, beta={beta})...')
        rankings = rerank_results(searcher, qdata, None, alpha=alpha, beta=beta, ncells=args.ncells, topk=args.topk)
        out_path = out_dir / f'rankings_{name}.json'
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(rankings, f)
        print(f'  Saved -> {out_path}')


if __name__ == '__main__':
    main()
