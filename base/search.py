"""
base/search.py
ColBERTv2 approximate search with IVF + residual decompression + MaxSim.
Outputs per-query candidate set including per-token MaxSim scores.
"""
import argparse
from pathlib import Path

import torch

from base.residual import dequantize_residuals, unpack_bits
from base.maxsim import maxsim_per_token


class ColBERTSearcher:
    def __init__(self, index_dir, nbits=2, device='cpu'):
        self.index_dir = Path(index_dir)
        self.nbits = nbits
        self.device = device

        self.centroids = torch.load(self.index_dir / 'centroids.pt', map_location='cpu').to(device)
        self.codes = torch.load(self.index_dir / 'codes.pt', map_location='cpu').to(device)
        self.ivf = torch.load(self.index_dir / 'ivf.pt')
        metadata = torch.load(self.index_dir / 'metadata.pt', map_location='cpu')
        self.doclens = metadata['doclens']
        self.pids = metadata['pids']

        self.offsets = [0]
        for length in self.doclens:
            self.offsets.append(self.offsets[-1] + length)
        self.offsets = torch.tensor(self.offsets, dtype=torch.long, device=device)

        residual_data = torch.load(self.index_dir / 'residuals.pt', map_location='cpu')
        self.bucket_weights = residual_data['bucket_weights'].to(device)
        if 'packed' in residual_data:
            self.residual_idx = unpack_bits(residual_data['packed'], (self.codes.shape[0], self.centroids.shape[1]), nbits=nbits).to(device)
        else:
            self.residual_idx = residual_data['idx'].to(device)

        self.K = self.centroids.shape[0]

    def reconstruct_candidates(self, candidate_pid_indices):
        starts = self.offsets[candidate_pid_indices]
        ends = self.offsets[torch.tensor([i + 1 for i in candidate_pid_indices], device=self.device)]

        token_indices = torch.cat([
            torch.arange(start.item(), end.item(), device=self.device)
            for start, end in zip(starts, ends)
        ])

        c = self.codes[token_indices]
        ridx = self.residual_idx[token_indices]
        emb = self.centroids[c] + dequantize_residuals(ridx, self.bucket_weights)
        emb = torch.nn.functional.normalize(emb, p=2, dim=-1)

        doclens_candidates = [self.doclens[i] for i in candidate_pid_indices]
        return emb, doclens_candidates

    def get_candidates(self, Q, ncells=4):
        scores = Q @ self.centroids.T
        top_cids = scores.topk(ncells, dim=-1, largest=True).indices
        top_cids = top_cids.flatten().unique().tolist()

        cand = set()
        for cid in top_cids:
            if cid in self.ivf:
                cand.update(self.ivf[cid])
        return sorted(list(cand))

    def search_query(self, Q, ncells=4, topk=100):
        Q = torch.nn.functional.normalize(Q, p=2, dim=-1)
        candidates = self.get_candidates(Q, ncells=ncells)

        if len(candidates) == 0:
            return {'pids': [], 'scores': [], 'token_scores': torch.zeros(0, Q.shape[0])}

        emb, doclens_cand = self.reconstruct_candidates(candidates)
        sim = Q @ emb.T

        token_scores = []
        scores = []
        offset = 0
        for length in doclens_cand:
            ts = sim[:, offset:offset+length].max(dim=-1).values
            token_scores.append(ts)
            scores.append(ts.sum().item())
            offset += length

        token_scores = torch.stack(token_scores, dim=0)
        scores = torch.tensor(scores, device=self.device)
        sorted_idx = scores.argsort(descending=True)[:topk]

        return {
            'pids': [self.pids[candidates[i]] for i in sorted_idx.tolist()],
            'scores': scores[sorted_idx].tolist(),
            'token_scores': token_scores[sorted_idx].cpu(),
        }

    def batch_search(self, Qs, qids, ncells=4, topk=100):
        results = {}
        for qid, Q in zip(qids, Qs):
            results[qid] = self.search_query(Q, ncells=ncells, topk=topk)
        return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--index_dir', default='outputs')
    parser.add_argument('--query_emb', default='outputs/query_embeddings.pt')
    parser.add_argument('--output_candidates', default='outputs/base_candidates.pt')
    parser.add_argument('--output_ranking', default='outputs/rankings_original.json')
    parser.add_argument('--ncells', type=int, default=4)
    parser.add_argument('--topk', type=int, default=100)
    parser.add_argument('--device', default=None)
    args = parser.parse_args()

    device = args.device or ('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f'Using device: {device}')

    searcher = ColBERTSearcher(args.index_dir, device=device)
    qdata = torch.load(args.query_emb, map_location='cpu')
    Qs = qdata['Q'].to(device)
    qids = qdata['qids']

    results = searcher.batch_search(Qs, qids, ncells=args.ncells, topk=args.topk)

    torch.save(results, args.output_candidates)
    print(f'Saved base candidates -> {args.output_candidates}')

    import json
    ranking = {qid: res['pids'] for qid, res in results.items()}
    with open(args.output_ranking, 'w', encoding='utf-8') as f:
        json.dump(ranking, f)
    print(f'Saved original ranking -> {args.output_ranking}')


if __name__ == '__main__':
    main()
