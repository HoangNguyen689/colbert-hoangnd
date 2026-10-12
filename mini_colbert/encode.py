"""
encode.py
Load the official ColBERTv2 checkpoint and encode SciFact corpus + queries.
Outputs:
  outputs/doc_embeddings.pt  -> {D: tensor(N_tokens, 128), doclens: list[int], pids: list[str]}
  outputs/query_embeddings.pt -> {Q: tensor(N_queries, 32, 128), qids: list[str], texts: list[str]}
"""
import os
import json
import argparse
from pathlib import Path

import torch
from tqdm import tqdm
from colbert.modeling.checkpoint import Checkpoint
from colbert.infra.config import ColBERTConfig


def load_scifact(data_root: str):
    """Load SciFact corpus, queries, qrels from BEIR format."""
    data_root = Path(data_root)
    corpus_file = data_root / 'scifact' / 'corpus.jsonl'
    queries_file = data_root / 'scifact' / 'queries.jsonl'
    qrels_file = data_root / 'scifact' / 'qrels' / 'test.tsv'

    if not corpus_file.exists() or not queries_file.exists():
        print('SciFact not found, downloading via BEIR...')
        from beir import util
        url = 'https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip'
        util.download_and_unzip(url, str(data_root))

    corpus = {}
    with open(corpus_file, 'r', encoding='utf-8') as f:
        for line in f:
            obj = json.loads(line)
            corpus[obj['_id']] = obj.get('title', '') + ' ' + obj.get('text', '')

    queries = {}
    with open(queries_file, 'r', encoding='utf-8') as f:
        for line in f:
            obj = json.loads(line)
            queries[obj['_id']] = obj['text']

    qrels = {}
    with open(qrels_file, 'r', encoding='utf-8') as f:
        next(f)  # skip header
        for line in f:
            qid, pid, score = line.strip().split('\t')
            qrels.setdefault(qid, {})[pid] = int(score)

    return corpus, queries, qrels


def load_checkpoint(checkpoint_name: str = 'colbert-ir/colbertv2.0'):
    config = ColBERTConfig(
        query_maxlen=32,
        doc_maxlen=180,
        dim=128,
        checkpoint=checkpoint_name,
        similarity='cosine',
    )
    cp = Checkpoint(checkpoint_name, colbert_config=config)
    cp.eval()
    return cp


def encode_corpus(cp, corpus, out_path, batch_size=32):
    pids = list(corpus.keys())
    docs = [corpus[pid] for pid in pids]

    all_D, all_doclens = [], []
    for i in tqdm(range(0, len(docs), batch_size), desc='Encoding docs'):
        batch = docs[i:i+batch_size]
        D, doclens = cp.docFromText(batch, bsize=batch_size, keep_dims='flatten')
        all_D.append(D.cpu())
        all_doclens.extend(doclens)

    D = torch.cat(all_D, dim=0)  # (N_tokens, 128)
    torch.save({
        'D': D,
        'doclens': all_doclens,
        'pids': pids,
    }, out_path)
    print(f'Saved corpus embeddings: {D.shape} tokens, {len(pids)} docs -> {out_path}')
    return D, all_doclens, pids


def encode_queries(cp, queries, out_path, batch_size=32):
    qids = list(queries.keys())
    texts = [queries[qid] for qid in qids]

    all_Q, all_masks = [], []
    for i in tqdm(range(0, len(texts), batch_size), desc='Encoding queries'):
        batch = texts[i:i+batch_size]
        Q = cp.queryFromText(batch)
        _, mask = cp.query_tokenizer.tensorize(batch)
        all_Q.append(Q.cpu())
        all_masks.append(mask.cpu())

    Q = torch.cat(all_Q, dim=0)
    masks = torch.cat(all_masks, dim=0)
    torch.save({
        'Q': Q,
        'masks': masks,
        'qids': qids,
        'texts': texts,
    }, out_path)
    print(f'Saved query embeddings: {Q.shape}, masks: {masks.shape} -> {out_path}')
    return Q, masks, qids, texts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', default='data')
    parser.add_argument('--output_dir', default='outputs')
    parser.add_argument('--doc_batch_size', type=int, default=32)
    parser.add_argument('--query_batch_size', type=int, default=32)
    parser.add_argument('--checkpoint', default='colbert-ir/colbertv2.0')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print('Loading checkpoint...')
    cp = load_checkpoint(args.checkpoint)

    print('Loading SciFact...')
    corpus, queries, qrels = load_scifact(args.data_root)

    doc_out = Path(args.output_dir) / 'doc_embeddings.pt'
    query_out = Path(args.output_dir) / 'query_embeddings.pt'

    encode_corpus(cp, corpus, doc_out, batch_size=args.doc_batch_size)
    encode_queries(cp, queries, query_out, batch_size=args.query_batch_size)

    qrels_out = Path(args.output_dir) / 'qrels_test.json'
    with open(qrels_out, 'w', encoding='utf-8') as f:
        json.dump(qrels, f)
    print(f'Saved qrels -> {qrels_out}')


if __name__ == '__main__':
    main()
