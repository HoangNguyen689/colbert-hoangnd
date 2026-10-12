"""
cad/idf.py
Compute corpus-level IDF for query token IDs.
IDF(t) = log((N + 1) / (df(t) + 1))
Special tokens (PAD, [Q]/[D], CLS, SEP, MASK) get neutral IDF = 1.0.
"""
import argparse
import json
import math
from pathlib import Path

import torch
from tqdm import tqdm

from base.encode import load_checkpoint, load_scifact


SPECIAL_IDS = {0, 1, 101, 102, 103}


def compute_idf(cp, corpus, query_token_ids):
    N = len(corpus)
    df = {}
    for text in tqdm(corpus.values(), desc='Computing corpus IDF'):
        input_ids, _ = cp.doc_tokenizer.tensorize([text])
        unique = set(input_ids[0].tolist()) - SPECIAL_IDS
        for tid in unique:
            df[tid] = df.get(tid, 0) + 1

    unique_query_ids = query_token_ids.unique().tolist()
    idf = {}
    for tid in unique_query_ids:
        if tid in SPECIAL_IDS:
            idf[tid] = 1.0
        else:
            idf[tid] = math.log((N + 1) / (df.get(tid, 0) + 1))
    return idf


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', default='data')
    parser.add_argument('--query_emb', default='outputs/query_embeddings.pt')
    parser.add_argument('--output', default='outputs/idf.pt')
    args = parser.parse_args()

    print('Loading checkpoint...')
    cp = load_checkpoint()
    print('Loading corpus...')
    corpus, _, _ = load_scifact(args.data_root)
    qdata = torch.load(args.query_emb, map_location='cpu')

    idf = compute_idf(cp, corpus, qdata['token_ids'])
    torch.save(idf, args.output)
    print(f'Saved IDF for {len(idf)} tokens -> {args.output}')


if __name__ == '__main__':
    main()
