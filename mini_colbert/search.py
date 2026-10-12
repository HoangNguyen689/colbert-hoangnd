"""
search.py
ColBERTv2 approximate search with IVF + residual decompression + MaxSim.
Vectorized over candidates for speed.
"""
import argparse
from pathlib import Path

import torch
import numpy as np

from mini_colbert.residual import dequantize_residuals, unpack_bits


class ColBERTSearcher:
    def __init__(self, index_dir, nbits=2, device='cpu'):
        self.index_dir = Path(index_dir)
        self.nbits = nbits
        self.device = device

        self.centroids = torch.load(self.index_dir / 'centroids.pt', map_location='cpu').to(device)
        self.codes = torch.load(self.index_dir / 'codes.pt', map_location='cpu').to(device)
        self.ivf = torch.load(self.index_dir / 'ivf.pt')
        emb = torch.load(self.index_dir / 'doc_embeddings.pt', map_location='cpu')
        self.doclens = emb['doclens']
        self.pids = emb['pids']

        # Build passage offset map
        self.offsets = [0]
        for length in self.doclens:
            self.offsets.append(self.offsets[-1] + length)
        self.offsets = torch.tensor(self.offsets, dtype=torch.long, device=device)

        # Load quantized residuals
        residual_data = torch.load(self.index_dir / 'residuals.pt', map_location='cpu')
        self.residual_lo = residual_data['lo'].to(device)
        self.residual_step = residual_data['step'].to(device)
        if 'packed' in residual_data:
            self.residual_idx = unpack_bits(residual_data['packed'], (self.codes.shape[0], self.centroids.shape[1]), nbits=nbits).to(device)
        else:
            self.residual_idx = residual_data['idx'].to(device)

        self.K = self.centroids.shape[0]
        self.dim = self.centroids.shape[1]

    def reconstruct_candidate_embeddings(self, candidate_pid_indices):
        """
        Vectorized reconstruction of all token embeddings for candidate passages.
        Returns (all_embeddings, doclens_candidates)
        """
        starts = self.offsets[candidate_pid_indices]
        ends = self.offsets[torch.tensor([i + 1 for i in candidate_pid_indices], device=self.device)]
        total_tokens = (ends - starts).sum().item()

        # Build global token indices
        token_indices = torch.cat([
            torch.arange(start.item(), end.item(), device=self.device)
            for start, end in zip(starts, ends)
        ])

        c = self.codes[token_indices]
        ridx = self.residual_idx[token_indices]
        emb = self.centroids[c] + dequantize_residuals(ridx, self.residual_lo, self.residual_step)
        emb = torch.nn.functional.normalize(emb, p=2, dim=-1)

        doclens_candidates = [self.doclens[i] for i in candidate_pid_indices]
        return emb, doclens_candidates

    def get_candidates(self, Q, ncells=4):
        """
        Q: (query_len, dim) tensor
        Returns list of passage indices that share at least one centroid with query tokens.
        """
        q_centroid_dists = torch.cdist(Q, self.centroids, p=2)  # (Q_len, K)
        top_cids = q_centroid_dists.topk(ncells, dim=-1, largest=False).indices  # (Q_len, ncells)
        top_cids = top_cids.flatten().unique().tolist()

        cand = set()
        for cid in top_cids:
            if cid in self.ivf:
                cand.update(self.ivf[cid])
        return sorted(list(cand))

    def score_query(self, Q, mask=None, ncells=4, topk=100):
        """
        Q: (query_len, dim) tensor
        mask: (query_len,) bool tensor; True for real query tokens
        Returns (pids, scores) sorted by descending MaxSim score.
        """
        Q = torch.nn.functional.normalize(Q, p=2, dim=-1)
        candidates = self.get_candidates(Q, ncells=ncells)

        if len(candidates) == 0:
            return [], []

        emb, doclens_cand = self.reconstruct_candidate_embeddings(candidates)
        sim = Q @ emb.T  # (query_len, total_tokens)
        if mask is not None:
            sim = sim[mask]

        scores = []
        offset = 0
        for length in doclens_cand:
            doc_sim = sim[:, offset:offset+length]
            scores.append(doc_sim.max(dim=-1).values.sum().item())
            offset += length

        scores = torch.tensor(scores, device=self.device)
        sorted_idx = scores.argsort(descending=True)[:topk]
        return [self.pids[candidates[i]] for i in sorted_idx.tolist()], scores[sorted_idx].tolist()

    def batch_search(self, Qs, qids, masks=None, ncells=4, topk=100):
        """
        Qs: (N_queries, query_len, dim) tensor
        qids: list[str]
        masks: (N_queries, query_len) bool tensor
        Returns dict qid -> {pids: [...], scores: [...]}
        """
        results = {}
        if masks is None:
            masks = [None] * len(qids)
        for qid, Q, mask in zip(qids, Qs, masks):
            pids, scores = self.score_query(Q, mask=mask, ncells=ncells, topk=topk)
            results[qid] = {'pids': pids, 'scores': scores}
        return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--index_dir', default='outputs')
    parser.add_argument('--query_emb', default='outputs/query_embeddings.pt')
    parser.add_argument('--output', default='outputs/search_results.pt')
    parser.add_argument('--ncells', type=int, default=4)
    parser.add_argument('--topk', type=int, default=100)
    parser.add_argument('--device', default=None, help='cpu/mps/cuda; default=mps if available else cpu')
    args = parser.parse_args()

    device = args.device or ('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f'Using device: {device}')

    searcher = ColBERTSearcher(args.index_dir, device=device)
    qdata = torch.load(args.query_emb, map_location='cpu')
    Qs = qdata['Q'].to(device)
    masks = qdata.get('masks', None)
    if masks is not None:
        masks = masks.to(device).bool()
    qids = qdata['qids']

    results = searcher.batch_search(Qs, qids, masks=masks, ncells=args.ncells, topk=args.topk)
    torch.save(results, args.output)
    print(f'Saved search results -> {args.output}')


if __name__ == '__main__':
    main()
