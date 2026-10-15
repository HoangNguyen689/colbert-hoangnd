"""
cad/experiment.py
Offline grid search for CAD-MaxSim -- no encoding and no GPU required.

Sweeps, over an existing base candidate pool (base_candidates.pt):

  * combine mode : mul  ->  w_i = IDF_i^a * (s_i + eps)^b     (original CAD)
                   add  ->  w_i = 1 + a*(IDF_i/mean - 1) + b*(s_i/mean - 1)
  * signal s     : std, mad, gap, margin, cv, snr, ldf   (see cad/signals.py)
  * grids of alpha and beta

reports nDCG@10 / MRR@10 / Recall@100 / nDCG@100 for every configuration, then
runs a paired bootstrap on per-query nDCG@10 against the IDF baseline so a win
can be claimed with a p-value instead of a raw decimal.

Inputs per dataset:
    --candidates  base_candidates.pt
    --query_emb   query_embeddings.pt   (or rerank_inputs.pt from cad.build_rerank_inputs)
    --idf         idf.pt
    --qrels       qrels_test.json

Example:
    python -m cad.build_rerank_inputs --dataset fiqa --output_dir outputs/rerank/fiqa
    python -m cad.experiment \
        --candidates outputs/rerank/fiqa/base_candidates.pt \
        --query_emb  outputs/rerank/fiqa/rerank_inputs.pt \
        --idf        outputs/rerank/fiqa/idf.pt \
        --qrels      outputs/rerank/fiqa/qrels_test.json
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from cad.signals import SIGNALS
from evaluation.metrics import dcg, evaluate, load_json

EPS = 1e-6
SPECIAL_IDS = {0, 1, 101, 102, 103}


# --------------------------------------------------------------------------- #
# weighting
# --------------------------------------------------------------------------- #
def make_weights(idf, sig, alpha, beta, mode='mul', valid=None):
    idf = idf.clamp(min=0.0)
    sig = sig.clamp(min=0.0)
    if mode == 'mul':
        w = (idf ** alpha) * ((sig + EPS) ** beta)
    elif mode == 'add':
        idn = idf / (idf.mean() + EPS)
        sn = sig / (sig.mean() + EPS)
        w = 1.0 + alpha * (idn - 1.0) + beta * (sn - 1.0)
        w = w.clamp(min=0.0)
    else:
        raise ValueError(f'unknown mode {mode}')
    denom = w[valid].mean() if (valid is not None and bool(valid.any())) else w.mean()
    return w / (denom + EPS)


# --------------------------------------------------------------------------- #
# metrics
# --------------------------------------------------------------------------- #
def per_query_ndcg(rankings, qrels, k=10):
    out = {}
    for qid, rel in qrels.items():
        gains = [rel.get(pid, 0) for pid in rankings.get(qid, [])]
        ideal = sorted(rel.values(), reverse=True)
        idcg = dcg(ideal, k)
        out[qid] = (dcg(gains, k) / idcg) if idcg > 0 else 0.0
    return out


def per_query_mrr(rankings, qrels, k=10):
    out = {}
    for qid, rel in qrels.items():
        rr = 0.0
        for i, pid in enumerate(rankings.get(qid, [])[:k]):
            if rel.get(pid, 0) > 0:
                rr = 1.0 / (i + 1)
                break
        out[qid] = rr
    return out


def paired_bootstrap(a, b, iters=2000, seed=0):
    """Paired bootstrap of mean(a - b) over the same queries."""
    d = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    n = len(d)
    rng = np.random.default_rng(seed)
    means = np.empty(iters)
    for i in range(iters):
        means[i] = d[rng.integers(0, n, n)].mean()
    lo, hi = np.percentile(means, [2.5, 97.5])
    p = 2.0 * min((means <= 0.0).mean(), (means >= 0.0).mean())
    return float(d.mean()), float(lo), float(hi), float(min(p, 1.0))


# --------------------------------------------------------------------------- #
# data
# --------------------------------------------------------------------------- #
def precompute(candidates_path, qdata_path, idf_path):
    cands = torch.load(candidates_path, map_location='cpu')
    qdata = torch.load(qdata_path, map_location='cpu')
    idf_dict = torch.load(idf_path, map_location='cpu')
    qids = qdata['qids']
    token_ids = qdata['token_ids']

    pre = []
    for i, qid in enumerate(qids):
        cand = cands.get(qid)
        if not cand or len(cand.get('candidate_pids', [])) == 0:
            continue
        ts = cand['token_scores'].float()
        ids = token_ids[i]
        idf_vals = torch.tensor(
            [idf_dict.get(int(t), 1.0) for t in ids.tolist()], dtype=torch.float32,
        )
        valid = ~torch.isin(ids.cpu(), torch.tensor(sorted(SPECIAL_IDS)))
        pre.append({
            'qid': qid,
            'ts': ts,
            'pids': cand['candidate_pids'],
            'idf': idf_vals,
            'valid': valid,
            'sig': {name: fn(ts) for name, fn in SIGNALS.items()},
        })
    return pre


def run_config(pre, mode, sig_name, alpha, beta, topk, valid_only):
    rankings = {}
    for r in pre:
        w = make_weights(r['idf'], r['sig'][sig_name], alpha, beta, mode,
                         valid=r['valid'] if valid_only else None)
        sc = (r['ts'] * w.unsqueeze(0)).sum(dim=1)
        order = torch.argsort(sc, descending=True)[:topk].tolist()
        rankings[r['qid']] = [r['pids'][i] for i in order]
    return rankings


def score_config(pre, qrels, mode, sig_name, alpha, beta, topk, valid_only):
    rankings = run_config(pre, mode, sig_name, alpha, beta, topk, valid_only)
    m = evaluate(rankings, qrels)
    ndcg100 = per_query_ndcg(rankings, qrels, k=100)
    mrr100 = per_query_mrr(rankings, qrels, k=100)
    m['nDCG@100'] = float(np.mean(list(ndcg100.values())))
    m['MRR@100'] = float(np.mean(list(mrr100.values())))
    return m, rankings


def parse_list(text, cast=float):
    return [cast(x) for x in str(text).split(',') if x.strip() != '']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--candidates', required=True)
    ap.add_argument('--query_emb', required=True)
    ap.add_argument('--idf', required=True)
    ap.add_argument('--qrels', required=True)
    ap.add_argument('--topk', type=int, default=100)
    ap.add_argument('--alphas', default='0,0.25,0.5,1,1.5,2')
    ap.add_argument('--betas', default='0,0.05,0.1,0.2,0.3,0.5,0.8,1')
    ap.add_argument('--signals', default='std,mad,gap,margin,cv,snr,ldf')
    ap.add_argument('--combines', default='mul,add')
    ap.add_argument('--valid-only', action='store_true',
                    help='normalise weights over real query tokens only (drop PAD/CLS/SEP)')
    ap.add_argument('--top-rows', type=int, default=20)
    ap.add_argument('--json-out', default=None)
    args = ap.parse_args()

    alphas = parse_list(args.alphas)
    betas = parse_list(args.betas)
    signals = [s for s in args.signals.split(',') if s]
    combines = [c for c in args.combines.split(',') if c]

    print('Loading inputs...')
    pre = precompute(args.candidates, args.query_emb, args.idf)
    qrels = load_json(args.qrels)
    print(f'{len(pre)} queries with candidates, {len(qrels)} queries in qrels')

    # ---- baselines -------------------------------------------------------- #
    base_m, base_rank = score_config(pre, qrels, 'mul', 'std', 0.0, 0.0, args.topk, args.valid_only)
    idf_m, idf_rank = score_config(pre, qrels, 'mul', 'std', 1.0, 0.0, args.topk, args.valid_only)

    base_pq = per_query_ndcg(base_rank, qrels, k=10)
    idf_pq = per_query_ndcg(idf_rank, qrels, k=10)

    def line(tag, m):
        print(f'{tag:<28} {m["nDCG@10"]:>8.4f} {m["MRR@10"]:>8.4f} '
              f'{m["Recall@100"]:>11.4f} {m["nDCG@100"]:>9.4f} {m["MRR@100"]:>8.4f}')

    print()
    print(f'{"Config":<28} {"nDCG@10":>8} {"MRR@10":>8} {"Recall@100":>11} {"nDCG@100":>9} {"MRR@100":>8}')
    print('-' * 78)
    line('baseline: MaxSim (a=0,b=0)', base_m)
    line('baseline: IDF (a=1,b=0)', idf_m)
    print('-' * 78)

    # ---- grid ------------------------------------------------------------- #
    results = []
    n_cfg = len(combines) * len(signals) * len(alphas) * len(betas)
    print(f'Running {n_cfg} configurations...')
    for mode in combines:
        for sig_name in signals:
            for alpha in alphas:
                for beta in betas:
                    if beta == 0.0 and sig_name != signals[0]:
                        continue          # signal is irrelevant when beta=0
                    if mode == 'mul' and alpha == 0.0 and beta == 0.0:
                        continue          # already the baseline
                    m, _ = score_config(pre, qrels, mode, sig_name, alpha, beta,
                                        args.topk, args.valid_only)
                    results.append({
                        'mode': mode, 'signal': sig_name, 'alpha': alpha, 'beta': beta,
                        **{k: float(v) for k, v in m.items()},
                    })

    ref = idf_m['nDCG@10']
    results.sort(key=lambda r: r['nDCG@10'], reverse=True)

    print()
    print(f'{"mode":<5}{"signal":<8}{"alpha":>7}{"beta":>7} {"nDCG@10":>8} {"MRR@10":>8} '
          f'{"Recall@100":>11} {"nDCG@100":>9} {"vs IDF":>9}')
    print('-' * 82)
    for r in results[:args.top_rows]:
        gain = (r['nDCG@10'] - ref) / ref * 100.0 if ref > 0 else 0.0
        print(f'{r["mode"]:<5}{r["signal"]:<8}{r["alpha"]:>7.2f}{r["beta"]:>7.2f} '
              f'{r["nDCG@10"]:>8.4f} {r["MRR@10"]:>8.4f} {r["Recall@100"]:>11.4f} '
              f'{r["nDCG@100"]:>9.4f} {gain:>8.2f}%')

    # ---- significance of the winner -------------------------------------- #
    if results:
        best = results[0]
        _, best_rank = score_config(pre, qrels, best['mode'], best['signal'],
                                    best['alpha'], best['beta'], args.topk, args.valid_only)
        best_pq = per_query_ndcg(best_rank, qrels, k=10)
        qids = list(qrels.keys())
        a = [best_pq.get(q, 0.0) for q in qids]
        b_idf = [idf_pq.get(q, 0.0) for q in qids]
        b_base = [base_pq.get(q, 0.0) for q in qids]

        d_i, lo_i, hi_i, p_i = paired_bootstrap(a, b_idf)
        d_b, lo_b, hi_b, p_b = paired_bootstrap(a, b_base)

        print()
        print('Best config:', best)
        print(f'nDCG@10 vs IDF   : delta={d_i:+.4f}  95% CI [{lo_i:+.4f}, {hi_i:+.4f}]  p={p_i:.4f}')
        print(f'nDCG@10 vs MaxSim: delta={d_b:+.4f}  95% CI [{lo_b:+.4f}, {hi_b:+.4f}]  p={p_b:.4f}')
        print('(p < 0.05 => khac biet co y nghia thong ke)')

    if args.json_out:
        Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.json_out, 'w', encoding='utf-8') as f:
            json.dump({'baselines': {'maxsim': base_m, 'idf': idf_m}, 'grid': results}, f, indent=2)
        print('Saved ->', args.json_out)


if __name__ == '__main__':
    main()
