"""
cad/tune_cad.py
Grid-search / fine-grained parameter tuning for CAD-MaxSim (alpha and beta).
Helps find the optimal balance between global corpus IDF and local candidate-adaptive StdDev.

Usage:
    python -m cad.tune_cad --output_dir outputs --qrels outputs/qrels_test.json
"""
import argparse
from pathlib import Path
import torch

from cad.rerank import rerank_all
from evaluation.metrics import evaluate, load_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--candidates', default='outputs/base_candidates.pt')
    parser.add_argument('--query_emb', default='outputs/query_embeddings.pt')
    parser.add_argument('--idf', default='outputs/idf.pt')
    parser.add_argument('--qrels', default='outputs/qrels_test.json')
    parser.add_argument('--output_dir', default='outputs')
    parser.add_argument('--topk', type=int, default=100)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    if not (out_dir / 'base_candidates.pt').exists():
        print(f"Error: {out_dir / 'base_candidates.pt'} not found. Please run base pipeline first.")
        return

    base_candidates = torch.load(args.candidates, map_location='cpu')
    qdata = torch.load(args.query_emb, map_location='cpu')
    idf_dict = torch.load(args.idf, map_location='cpu')
    qrels = load_json(args.qrels)

    qids = qdata['qids']
    token_ids = qdata['token_ids']

    # Grid search candidate values
    alpha_list = [1.0]
    beta_list = [0.0, 0.1, 0.2, 0.3, 0.5, 0.8, 1.0]

    print(f"{'Config':<20} {'Alpha':<8} {'Beta':<8} {'nDCG@10':>10} {'MRR@10':>10} {'Recall@100':>12}")
    print('-' * 72)

    best_ndcg = -1.0
    best_config = None

    for alpha in alpha_list:
        for beta in beta_list:
            rankings = rerank_all(base_candidates, idf_dict, token_ids, qids, alpha=alpha, beta=beta)
            trimmed_rankings = {qid: pids[:args.topk] for qid, pids in rankings.items()}
            metrics = evaluate(trimmed_rankings, qrels)

            label = f"a={alpha:.1f}, b={beta:.1f}"
            print(f"{label:<20} {alpha:<8.1f} {beta:<8.1f} {metrics['nDCG@10']:>10.4f} {metrics['MRR@10']:>10.4f} {metrics['Recall@100']:>12.4f}")

            if metrics['nDCG@10'] > best_ndcg:
                best_ndcg = metrics['nDCG@10']
                best_config = (alpha, beta, metrics)

    if best_config:
        b_a, b_b, b_m = best_config
        print('-' * 72)
        print(f"Best configuration: Alpha={b_a}, Beta={b_b} => nDCG@10 = {b_m['nDCG@10']:.4f}, MRR@10 = {b_m['MRR@10']:.4f}")


if __name__ == '__main__':
    main()
