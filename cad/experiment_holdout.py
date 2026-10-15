"""
cad/experiment_holdout.py
Held-out check for the CAD-Sim grid search.

Why this file exists
--------------------
cad/experiment.py sweeps the grid and picks the best config on the *same* queries
it reports the gain on. That makes the reported gain optimistic: the config is
chosen after seeing the test queries, and the paired bootstrap does not correct
for that choice.

This script runs the honest version, as its own program so that
cad/experiment.py stays a plain full-set sweep:

  1. split the evaluated queries into two disjoint halves, A and B;
  2. sweep the grid on A only and pick the best config by nDCG@10;
  3. score that single config on B and compare with the IDF baseline on B using
     a paired bootstrap;
  4. also report the best config on B itself, as an oracle upper bound on what
     selection could reach there.

Steps 2-4 are repeated over several seeds and averaged. The gap between the
held-out gain and the oracle gain is the part of the earlier improvement that
came from picking the config rather than from the weighting.

Usage
-----
    python -m cad.experiment_holdout \
        --candidates outputs/rerank/<ds>/base_candidates.pt \
        --query_emb  outputs/rerank/<ds>/rerank_inputs.pt \
        --idf        outputs/rerank/<ds>/idf.pt \
        --qrels      outputs/rerank/<ds>/qrels_test.json \
        --seeds 0,1,2 \
        --json-out   <drive>/colbert-run/<ds>/holdout_results.json

The helpers below intentionally mirror cad/experiment.py, so that this file can
be read top to bottom without jumping to another module.
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
# weighting and metrics (same definitions as cad/experiment.py)
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
        if not cand:
            continue
        pids = cand.get('candidate_pids', cand.get('pids'))
        ts_raw = cand.get('token_scores')
        if pids is None or ts_raw is None or len(pids) == 0:
            continue
        ts = ts_raw.float()
        ids = token_ids[i]
        idf_vals = torch.tensor(
            [idf_dict.get(int(t), 1.0) for t in ids.tolist()], dtype=torch.float32,
        )
        valid = ~torch.isin(ids.cpu(), torch.tensor(sorted(SPECIAL_IDS)))
        pre.append({
            'qid': qid,
            'ts': ts,
            'pids': pids,
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


# --------------------------------------------------------------------------- #
# grid and split
# --------------------------------------------------------------------------- #
def parse_list(text, cast=float):
    return [cast(x) for x in str(text).split(',') if x.strip() != '']


def iter_configs(combines, signals, alphas, betas):
    """Every configuration of the sweep, in a fixed order."""
    for mode in combines:
        for sig_name in signals:
            for alpha in alphas:
                for beta in betas:
                    if beta == 0.0 and sig_name != signals[0]:
                        continue          # signal is irrelevant when beta=0
                    if mode == 'mul' and alpha == 0.0 and beta == 0.0:
                        continue          # already the baseline
                    yield mode, sig_name, alpha, beta


def sweep(pre, qrels, combos, topk, valid_only):
    """Score every configuration, return the rows unsorted."""
    rows = []
    for mode, sig_name, alpha, beta in combos:
        m, _ = score_config(pre, qrels, mode, sig_name, alpha, beta, topk, valid_only)
        rows.append({
            'mode': mode, 'signal': sig_name, 'alpha': alpha, 'beta': beta,
            **{k: float(v) for k, v in m.items()},
        })
    return rows


def split_qids(qids, frac, seed):
    """Deterministic split of the evaluated qids into two disjoint halves."""
    rng = np.random.default_rng(seed)
    ids = sorted(qids)
    perm = rng.permutation(len(ids))
    n_a = int(round(len(ids) * frac))
    a = {ids[i] for i in perm[:n_a]}
    return a, set(ids) - a


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
    ap.add_argument('--valid-only', action='store_true')
    ap.add_argument('--seeds', default='0,1,2')
    ap.add_argument('--frac', type=float, default=0.5,
                    help='fraction of queries used for picking the config')
    ap.add_argument('--json-out', default=None)
    args = ap.parse_args()

    alphas = parse_list(args.alphas)
    betas = parse_list(args.betas)
    signals = [s for s in args.signals.split(',') if s]
    combines = [c for c in args.combines.split(',') if c]
    seeds = parse_list(args.seeds, int)
    combos = list(iter_configs(combines, signals, alphas, betas))

    print('Loading inputs...')
    pre = precompute(args.candidates, args.query_emb, args.idf)
    qrels = load_json(args.qrels)
    print(f'{len(pre)} queries with candidates, {len(qrels)} queries in qrels '
          f'({len(combos)} configurations)')

    # full-set baselines, so they can be checked against the earlier sweep
    base_m, _ = score_config(pre, qrels, 'mul', 'std', 0.0, 0.0, args.topk, args.valid_only)
    idf_m, _ = score_config(pre, qrels, 'mul', 'std', 1.0, 0.0, args.topk, args.valid_only)
    print()
    print(f'{"Config":<28} {"nDCG@10":>8} {"MRR@10":>8} {"Recall@100":>11} '
          f'{"nDCG@100":>9} {"MRR@100":>8}')
    print('-' * 78)
    for tag, m in (('baseline: MaxSim (a=0,b=0)', base_m), ('baseline: IDF (a=1,b=0)', idf_m)):
        print(f'{tag:<28} {m["nDCG@10"]:>8.4f} {m["MRR@10"]:>8.4f} '
              f'{m["Recall@100"]:>11.4f} {m["nDCG@100"]:>9.4f} {m["MRR@100"]:>8.4f}')

    print()
    print(f'Hold-out: pick the config on {args.frac:.0%} of the queries (A), '
          f'report on the other {1 - args.frac:.0%} (B).')
    print(f'{"seed":>5} {"nA":>6} {"nB":>6}  {"picked on A":<22} '
          f'{"nDCG@10 B":>10} {"IDF B":>7} {"gain":>8} {"p":>8} '
          f'{"oracle B":>9} {"gain":>8}')
    print('-' * 110)

    runs = []
    for seed in seeds:
        A, B = split_qids(list(qrels.keys()), args.frac, seed)
        preA = [r for r in pre if r['qid'] in A]
        preB = [r for r in pre if r['qid'] in B]
        qA = {q: qrels[q] for q in qrels if q in A}
        qB = {q: qrels[q] for q in qrels if q in B}
        if not preA or not preB or not qA or not qB:
            print(f'{seed:>5}  skipped: empty half')
            continue

        bestA = max(sweep(preA, qA, combos, args.topk, args.valid_only),
                    key=lambda r: r['nDCG@10'])
        m_B, rank_B = score_config(preB, qB, bestA['mode'], bestA['signal'],
                                   bestA['alpha'], bestA['beta'],
                                   args.topk, args.valid_only)
        idfB_m, idfB_rank = score_config(preB, qB, 'mul', 'std', 1.0, 0.0,
                                         args.topk, args.valid_only)

        qids_B = list(qB.keys())
        pq_best = per_query_ndcg(rank_B, qB, k=10)
        pq_idf = per_query_ndcg(idfB_rank, qB, k=10)
        p_idf = paired_bootstrap([pq_best.get(q, 0.0) for q in qids_B],
                                 [pq_idf.get(q, 0.0) for q in qids_B])[3]

        oracle_B = max(sweep(preB, qB, combos, args.topk, args.valid_only),
                       key=lambda r: r['nDCG@10'])['nDCG@10']

        idfB_ndcg = idfB_m['nDCG@10']
        gain = (m_B['nDCG@10'] - idfB_ndcg) / idfB_ndcg * 100.0 if idfB_ndcg > 0 else 0.0
        ogain = (oracle_B - idfB_ndcg) / idfB_ndcg * 100.0 if idfB_ndcg > 0 else 0.0
        cfg = f"{bestA['mode']} {bestA['signal']} {bestA['alpha']}/{bestA['beta']}"

        print(f'{seed:>5} {len(qA):>6} {len(qB):>6}  {cfg:<22} '
              f'{m_B["nDCG@10"]:>10.4f} {idfB_ndcg:>7.4f} {gain:>7.2f}% '
              f'{p_idf:>8.4f} {oracle_B:>9.4f} {ogain:>7.2f}%')
        runs.append({'seed': seed, 'nA': len(qA), 'nB': len(qB), 'config': cfg,
                     'ndcg_b': m_B['nDCG@10'], 'ndcg_b_idf': idfB_ndcg,
                     'gain_pct': gain, 'p_vs_idf': p_idf,
                     'oracle_b': oracle_B, 'oracle_gain_pct': ogain})

    report = None
    if runs:
        mean_gain = float(np.mean([r['gain_pct'] for r in runs]))
        mean_oracle = float(np.mean([r['oracle_gain_pct'] for r in runs]))
        n_sig = int(sum(1 for r in runs if r['p_vs_idf'] < 0.05))
        print('-' * 110)
        print(f'mean held-out gain vs IDF {mean_gain:+.2f}%   '
              f'mean oracle-on-B gain {mean_oracle:+.2f}%   '
              f'significant p<0.05 in {n_sig}/{len(runs)} splits')
        print('(the gap between the two means is the part of the earlier gain '
              'that came from picking the config)')
        report = {'frac': args.frac, 'seeds': seeds, 'runs': runs,
                  'mean_gain_pct': mean_gain,
                  'mean_oracle_gain_pct': mean_oracle,
                  'n_significant': n_sig}

    if args.json_out:
        Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.json_out, 'w', encoding='utf-8') as f:
            json.dump({'baselines': {'maxsim': base_m, 'idf': idf_m},
                       'split': report}, f, indent=2)
        print('Saved ->', args.json_out)


if __name__ == '__main__':
    main()
