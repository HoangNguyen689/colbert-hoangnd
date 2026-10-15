"""
base/search.py
ColBERTv2 approximate search with IVF + residual decompression + MaxSim.
Outputs per-query candidate set including per-token MaxSim scores.
"""
import argparse
from pathlib import Path

import torch
from tqdm import tqdm

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
        """Gather the token embeddings of every candidate passage (vectorised).

        Returns:
            emb         -> (total_tokens, dim) L2-normalised embeddings
            cand_of_tok -> (total_tokens,) index of the owning candidate
            lengths     -> (num_candidates,) token count per candidate
        """
        lengths = torch.tensor(
            [self.doclens[i] for i in candidate_pid_indices.tolist()],
            dtype=torch.long, device=self.device,
        )
        starts = self.offsets[candidate_pid_indices]
        total = int(lengths.sum().item())

        # Token -> global index, built without a Python loop over candidates.
        cum = torch.cumsum(lengths, 0) - lengths
        arange = torch.arange(total, device=self.device)
        token_indices = (torch.repeat_interleave(starts, lengths)
                         + (arange - torch.repeat_interleave(cum, lengths)))
        cand_of_tok = torch.repeat_interleave(
            torch.arange(candidate_pid_indices.numel(), device=self.device), lengths
        )

        c = self.codes[token_indices]
        ridx = self.residual_idx[token_indices]
        emb = self.centroids[c] + dequantize_residuals(ridx, self.bucket_weights)
        emb = torch.nn.functional.normalize(emb, p=2, dim=-1)
        return emb, cand_of_tok, lengths

    def get_candidates(self, Q, ncells=4):
        scores = Q @ self.centroids.T
        top_cids = scores.topk(ncells, dim=-1, largest=True).indices
        top_cids = top_cids.flatten().unique().tolist()

        cand = set()
        for cid in top_cids:
            if cid in self.ivf:
                cand.update(self.ivf[cid])
        return sorted(list(cand))

    def search_query(self, Q, ncells=4, candidate_topn=200, output_topk=100):
        Q = torch.nn.functional.normalize(Q, p=2, dim=-1)
        candidates = self.get_candidates(Q, ncells=ncells)

        if len(candidates) == 0:
            return {
                'candidate_pids': [],
                'candidate_scores': [],
                'token_scores': torch.zeros(0, Q.shape[0]),
            }

        candidate_pid_indices = torch.tensor(candidates, dtype=torch.long, device=self.device)
        emb, cand_of_tok, _ = self.reconstruct_candidates(candidate_pid_indices)
        sim = Q @ emb.T                       # (query_tokens, total_tokens)
        q_tokens = Q.shape[0]

        # Per-candidate MaxSim: for each query token take the max over that
        # candidate's tokens, then sum over query tokens. One scatter does the
        # whole thing; the previous per-candidate Python loop performed a GPU
        # sync per candidate and dominated the runtime on large datasets.
        token_scores = torch.full(
            (candidate_pid_indices.numel(), q_tokens), float('-inf'),
            dtype=sim.dtype, device=self.device,
        )
        token_scores.scatter_reduce_(
            0, cand_of_tok.unsqueeze(1).expand(-1, q_tokens), sim.T, reduce='amax',
        )
        scores = token_scores.sum(dim=1)

        # Keep a larger candidate pool for rerankers, then return top-k.
        sorted_idx = scores.argsort(descending=True)[:candidate_topn]

        return {
            'candidate_pids': [self.pids[int(p)] for p in candidate_pid_indices[sorted_idx].tolist()],
            'candidate_scores': scores[sorted_idx].tolist(),
            'token_scores': token_scores[sorted_idx].cpu(),
        }

    def batch_search(self, Qs, qids, ncells=4, candidate_topn=200, output_topk=100):
        results = {}
        for qid, Q in tqdm(zip(qids, Qs), total=len(qids), desc='Searching queries'):
            results[qid] = self.search_query(
                Q, ncells=ncells, candidate_topn=candidate_topn, output_topk=output_topk
            )
        return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--index_dir', default='outputs')
    parser.add_argument('--query_emb', default='outputs/query_embeddings.pt')
    parser.add_argument('--output_candidates', default='outputs/base_candidates.pt')
    parser.add_argument('--output_ranking', default='outputs/rankings_original.json')
    parser.add_argument('--ncells', type=int, default=4)
    parser.add_argument('--candidate_topn', type=int, default=200,
                        help='number of candidates to keep for rerankers')
    parser.add_argument('--output_topk', type=int, default=100,
                        help='number of results to output in the final ranking')
    parser.add_argument('--device', default=None)
    args = parser.parse_args()

    device = args.device or ('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f'Using device: {device}')

    searcher = ColBERTSearcher(args.index_dir, device=device)
    qdata = torch.load(args.query_emb, map_location='cpu')
    Qs = qdata['Q'].to(device)
    qids = qdata['qids']

    results = searcher.batch_search(
        Qs, qids, ncells=args.ncells,
        candidate_topn=args.candidate_topn, output_topk=args.output_topk
    )

    torch.save(results, args.output_candidates)
    print(f'Saved base candidates -> {args.output_candidates}')

    import json
    ranking = {qid: res['candidate_pids'][:args.output_topk] for qid, res in results.items()}
    with open(args.output_ranking, 'w', encoding='utf-8') as f:
        json.dump(ranking, f)
    print(f'Saved original ranking -> {args.output_ranking}')


if __name__ == '__main__':
    main()
