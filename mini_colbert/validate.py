"""
validate.py
Validate approximate MaxSim against exact MaxSim on original uncompressed embeddings.
"""
import argparse
import json
from pathlib import Path

import torch
import numpy as np
from scipy.stats import kendalltau

from mini_colbert.search import ColBERTSearcher


def exact_maxsim(Q, D, doclens, pids, query_mask=None):
    """
    Q: (Q_len, dim)
    D: (N_tokens, dim)
    returns dict pid -> score
    """
    Q = torch.nn.functional.normalize(Q, p=2, dim=-1)
    D = torch.nn.functional.normalize(D, p=2, dim=-1)
    sim = Q @ D.T  # (Q_len, N_tokens)

    if query_mask is not None:
        sim = sim[query_mask]

    scores = {}
    offset = 0
    for pid, length in zip(pids, doclens):
        doc_sim = sim[:, offset:offset+length]
        scores[pid] = doc_sim.max(dim=-1).values.sum().item()
        offset += length
    return scores


def approximate_maxsim(searcher, Q, mask=None, ncells=4):
    """Return dict pid -> score from approximate search."""
    Q = torch.nn.functional.normalize(Q, p=2, dim=-1)
    candidates = searcher.get_candidates(Q, ncells=ncells)
    scores = {}
    if len(candidates) == 0:
        return scores
    emb, doclens_cand = searcher.reconstruct_candidate_embeddings(candidates)
    sim = Q @ emb.T
    if mask is not None:
        sim = sim[mask]
    offset = 0
    for pid_idx, length in zip(candidates, doclens_cand):
        doc_sim = sim[:, offset:offset+length]
        scores[searcher.pids[pid_idx]] = doc_sim.max(dim=-1).values.sum().item()
        offset += length
    return scores


def compare_rankings(exact_scores, approx_scores, topk=10):
    exact_sorted = sorted(exact_scores.items(), key=lambda x: x[1], reverse=True)
    approx_sorted = sorted(approx_scores.items(), key=lambda x: x[1], reverse=True)

    exact_top = [pid for pid, _ in exact_sorted[:topk]]
    approx_top = [pid for pid, _ in approx_sorted[:topk]]

    overlap = len(set(exact_top) & set(approx_top))
    print(f'Top-{topk} overlap: {overlap}/{topk}')

    # Score correlation on common candidates
    common = set(exact_scores.keys()) & set(approx_scores.keys())
    if len(common) > 1:
        e = [exact_scores[pid] for pid in common]
        a = [approx_scores[pid] for pid in common]
        tau, _ = kendalltau(e, a)
        print(f'Kendall tau on {len(common)} candidates: {tau:.4f}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--index_dir', default='outputs')
    parser.add_argument('--query_emb', default='outputs/query_embeddings.pt')
    parser.add_argument('--n_queries', type=int, default=10)
    parser.add_argument('--ncells', type=int, default=4)
    parser.add_argument('--device', default=None)
    args = parser.parse_args()

    device = args.device or ('mps' if torch.backends.mps.is_available() else 'cpu')
    searcher = ColBERTSearcher(args.index_dir, device=device)
    qdata = torch.load(args.query_emb, map_location='cpu')
    emb = torch.load(args.index_dir / Path('doc_embeddings.pt'), map_location='cpu')
    D = emb['D']
    masks = qdata.get('masks', None)

    qids = qdata['qids'][:args.n_queries]
    Qs = qdata['Q'][:args.n_queries]
    if masks is not None:
        masks = masks[:args.n_queries]

    for i, (qid, Q) in enumerate(zip(qids, Qs)):
        print(f'\nQuery: {qid}')
        mask = masks[i].bool() if masks is not None else None
        exact_scores = exact_maxsim(Q, D, emb['doclens'], emb['pids'], query_mask=mask)
        approx_scores = approximate_maxsim(searcher, Q.to(device), mask=mask, ncells=args.ncells)
        compare_rankings(exact_scores, approx_scores, topk=10)


if __name__ == '__main__':
    main()
