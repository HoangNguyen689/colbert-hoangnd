"""
base/encode.py
Load official ColBERTv2 checkpoint and encode a BEIR dataset (corpus + queries).
Outputs:
  outputs/doc_embeddings.pt  -> {D: tensor, doclens: list, pids: list}
  outputs/metadata.pt        -> {doclens: list, pids: list}
  outputs/query_embeddings.pt -> {Q: tensor, token_ids: tensor, qids: list, texts: list}
"""
import os
import json
import argparse
import zipfile
from pathlib import Path

import torch
from tqdm import tqdm
from colbert.modeling.checkpoint import Checkpoint
from colbert.infra.config import ColBERTConfig

from base.datasets import BEIR_DATASETS, BEIR_URL


def _drop_corrupt_archive(data_root: Path, dataset: str):
    """BEIR's download_and_unzip skips the download when the archive already
    exists, so a partial/corrupt zip left by an interrupted run would make the
    unzip fail. Validate it and delete it so it is fetched again."""
    zip_path = data_root / f'{dataset}.zip'
    if not zip_path.exists():
        return
    try:
        with zipfile.ZipFile(zip_path) as zf:
            if zf.testzip() is not None:
                raise zipfile.BadZipFile('CRC check failed')
    except zipfile.BadZipFile:
        print(f'Corrupt archive {zip_path}, re-downloading...')
        zip_path.unlink()


def load_beir(data_root: str, dataset: str = 'scifact'):
    """Load corpus, queries, qrels from a BEIR-format dataset."""
    if dataset not in BEIR_DATASETS:
        raise ValueError(f'Unsupported dataset: {dataset}. Choose from {BEIR_DATASETS}')
    data_root = Path(data_root)
    corpus_file = data_root / dataset / 'corpus.jsonl'
    queries_file = data_root / dataset / 'queries.jsonl'
    qrels_file = data_root / dataset / 'qrels' / 'test.tsv'

    if not corpus_file.exists() or not queries_file.exists():
        print(f'{dataset} not found, downloading via BEIR...')
        _drop_corrupt_archive(data_root, dataset)
        from beir import util
        util.download_and_unzip(BEIR_URL.format(dataset), str(data_root))

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


def load_scifact(data_root: str):
    """Backward-compatible alias for load_beir(data_root, 'scifact')."""
    return load_beir(data_root, 'scifact')


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


def encode_corpus(cp, corpus, out_path, metadata_path, batch_size=32):
    pids = list(corpus.keys())
    docs = [corpus[pid] for pid in pids]

    all_D, all_doclens = [], []
    for i in tqdm(range(0, len(docs), batch_size), desc='Encoding docs'):
        batch = docs[i:i+batch_size]
        D, doclens = cp.docFromText(batch, bsize=batch_size, keep_dims='flatten')
        all_D.append(D.cpu())
        all_doclens.extend(doclens)

    D = torch.cat(all_D, dim=0)
    torch.save({'D': D, 'doclens': all_doclens, 'pids': pids}, out_path)
    torch.save({'doclens': all_doclens, 'pids': pids}, metadata_path)
    print(f'Saved corpus embeddings: {D.shape} tokens, {len(pids)} docs -> {out_path}')
    return D, all_doclens, pids


def encode_queries(cp, queries, out_path, batch_size=32):
    qids = list(queries.keys())
    texts = [queries[qid] for qid in qids]

    all_Q, all_token_ids = [], []
    for i in tqdm(range(0, len(texts), batch_size), desc='Encoding queries'):
        batch = texts[i:i+batch_size]
        Q = cp.queryFromText(batch)
        input_ids, _ = cp.query_tokenizer.tensorize(batch)
        all_Q.append(Q.cpu())
        all_token_ids.append(input_ids.cpu())

    Q = torch.cat(all_Q, dim=0)
    token_ids = torch.cat(all_token_ids, dim=0)
    torch.save({'Q': Q, 'token_ids': token_ids, 'qids': qids, 'texts': texts}, out_path)
    print(f'Saved query embeddings: {Q.shape}, token_ids: {token_ids.shape} -> {out_path}')
    return Q, token_ids, qids, texts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', default='data')
    parser.add_argument('--dataset', default='scifact', choices=BEIR_DATASETS)
    parser.add_argument('--output_dir', default='outputs')
    parser.add_argument('--doc_batch_size', type=int, default=32)
    parser.add_argument('--query_batch_size', type=int, default=32)
    parser.add_argument('--checkpoint', default='colbert-ir/colbertv2.0')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print('Loading checkpoint...')
    cp = load_checkpoint(args.checkpoint)

    print(f'Loading {args.dataset}...')
    corpus, queries, qrels = load_beir(args.data_root, args.dataset)

    doc_out = Path(args.output_dir) / 'doc_embeddings.pt'
    metadata_out = Path(args.output_dir) / 'metadata.pt'
    query_out = Path(args.output_dir) / 'query_embeddings.pt'

    encode_corpus(cp, corpus, doc_out, metadata_out, batch_size=args.doc_batch_size)
    encode_queries(cp, queries, query_out, batch_size=args.query_batch_size)

    qrels_out = Path(args.output_dir) / 'qrels_test.json'
    with open(qrels_out, 'w', encoding='utf-8') as f:
        json.dump(qrels, f)
    print(f'Saved qrels -> {qrels_out}')


if __name__ == '__main__':
    main()
