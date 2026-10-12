"""
cad_maxsim.py
Context-Aware Diversified MaxSim (CAD-MaxSim) weighting.

Given per-query-token MaxSim scores c_i(d) over a candidate set C:
  IDF_i      = corpus-level inverse document frequency of query token i
  StdDev_i   = standard deviation of c_i(d) over candidate set C
  w_i        = IDF_i^alpha * (StdDev_i + eps)^beta
  score(d)   = sum_i (w_i / mean(w)) * c_i(d)

Ablations:
  Original  (alpha=0, beta=0):  w_i = 1
  IDF       (alpha=1, beta=0):  w_i = IDF_i
  Adaptive  (alpha=0, beta=1):  w_i = StdDev_i + eps
  CAD       (alpha=1, beta=1):  w_i = IDF_i * (StdDev_i + eps)

Implementation note:
  CAD reranks the SAME top-N candidates produced by Original MaxSim,
  so comparison is fair: same index, same candidates, same embeddings,
  only the aggregation formula changes.
"""
import argparse
import json
from pathlib import Path

import torch
import numpy as np

from mini_colbert.search import ColBERTSearcher


def cad_weights(idf, std, alpha=0.0, beta=0.0, eps=1e-6):
    """
    idf: (Q_len,) tensor
    std: (Q_len,) tensor
    Returns weights (Q_len,) with mean = 1 and all positive.
    """
    idf = torch.clamp(idf, min=0.0)
    std = torch.clamp(std, min=0.0)
    weights = (idf ** alpha) * ((std + eps) ** beta)
    weights = weights / (weights.mean() + eps)
    return weights


def rerank_topn(searcher, qdata, idf_dict, alpha=0.0, beta=0.0, ncells=4, topn=100, rerank_topk=None):
    """
    Rerank top-N candidates from Original MaxSim using CAD weights.
    If alpha=beta=0, returns Original MaxSim ranking (no re-weighting).
    Returns rankings dict: qid -> list of pids.
    """
    device = searcher.centroids.device
    Qs = qdata['Q'].to(device)
    token_ids = qdata['token_ids']
    qids = qdata['qids']
    rerank_topk = rerank_topk or topn

    rankings = {}
    for idx, (qid, Q) in enumerate(zip(qids, Qs)):
        Q = torch.nn.functional.normalize(Q, p=2, dim=-1)
        candidates = searcher.get_candidates(Q, ncells=ncells)
        if len(candidates) == 0:
            rankings[qid] = []
            continue

        # 1. Original MaxSim on full candidate set
        emb, doclens_cand = searcher.reconstruct_candidate_embeddings(candidates)
        sim = Q @ emb.T  # (Q_len, total_tokens)

        orig_scores = []
        offset = 0
        for length in doclens_cand:
            doc_sim = sim[:, offset:offset+length]
            orig_scores.append(doc_sim.max(dim=-1).values.sum().item())
            offset += length
        orig_scores = torch.tensor(orig_scores, device=device)

        # Keep top-N for CAD reranking
        topn_idx = orig_scores.argsort(descending=True)[:topn]
        topn_candidates = [candidates[i] for i in topn_idx.tolist()]

        if alpha == 0.0 and beta == 0.0:
            # Original: return top-K from original scores
            rankings[qid] = [searcher.pids[c] for c in topn_candidates[:rerank_topk]]
            continue

        # 2. Reconstruct and score ONLY top-N candidates for CAD
        emb_top, doclens_top = searcher.reconstruct_candidate_embeddings(topn_candidates)
        sim_top = Q @ emb_top.T

        token_scores = []
        offset = 0
        for length in doclens_top:
            doc_sim = sim_top[:, offset:offset+length]
            token_scores.append(doc_sim.max(dim=-1).values)
            offset += length
        token_scores = torch.stack(token_scores, dim=1)  # (Q_len, topn)

        # 3. IDF and StdDev
        qids_tokens = token_ids[idx].tolist()
        idf_vals = torch.tensor([idf_dict.get(tid, 1.0) for tid in qids_tokens], device=device, dtype=torch.float32)
        std_vals = token_scores.std(dim=1)

        weights = cad_weights(idf_vals, std_vals, alpha=alpha, beta=beta)

        # 4. Re-score
        weighted_scores = (weights.unsqueeze(1) * token_scores).sum(dim=0)
        sorted_idx = weighted_scores.argsort(descending=True)
        top_pids = [searcher.pids[topn_candidates[i]] for i in sorted_idx[:rerank_topk].tolist()]
        rankings[qid] = top_pids

    return rankings


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--index_dir', default='outputs')
    parser.add_argument('--query_emb', default='outputs/query_embeddings.pt')
    parser.add_argument('--idf', default='outputs/idf.pt')
    parser.add_argument('--qrels', default='outputs/qrels_test.json')
    parser.add_argument('--output_dir', default='outputs')
    parser.add_argument('--ncells', type=int, default=4)
    parser.add_argument('--topn', type=int, default=100, help='Top-N Original candidates for CAD to rerank')
    parser.add_argument('--topk', type=int, default=100, help='Final ranking length')
    parser.add_argument('--device', default=None)
    args = parser.parse_args()

    device = args.device or ('mps' if torch.backends.mps.is_available() else 'cpu')
    searcher = ColBERTSearcher(args.index_dir, device=device)
    qdata = torch.load(args.query_emb, map_location='cpu')
    idf_dict = torch.load(args.idf, map_location='cpu')

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
        rankings = rerank_topn(searcher, qdata, idf_dict, alpha=alpha, beta=beta,
                               ncells=args.ncells, topn=args.topn, rerank_topk=args.topk)
        out_path = out_dir / f'rankings_{name}.json'
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(rankings, f)
        print(f'  Saved -> {out_path}')


if __name__ == '__main__':
    main()
