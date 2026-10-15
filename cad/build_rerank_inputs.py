"""
cad/build_rerank_inputs.py
Rebuild the two small inputs needed to rerank / tune CAD, so you never have to
re-encode a dataset:

  <output_dir>/idf.pt            -> {token_id: corpus IDF}   (same as cad.idf)
  <output_dir>/rerank_inputs.pt  -> {'qids': [...], 'token_ids': tensor}
                                    (same layout as query_embeddings.pt)

Only the tokenizer is loaded -- no model weights, no GPU, no forward pass. The
token ids are verified id-for-id against Checkpoint's query_tokenizer. The corpus
is the BEIR one under <data_root>/<dataset>, downloaded on first use.

Usage:
    python -m cad.build_rerank_inputs --dataset fiqa --output_dir outputs/rerank/fiqa
"""
import argparse
from pathlib import Path
from types import SimpleNamespace

import torch

from base.encode import BEIR_DATASETS, load_beir
from cad.idf import compute_idf


def load_tokenizers(checkpoint='colbert-ir/colbertv2.0', query_maxlen=32, doc_maxlen=180):
    """Same tokenizers Checkpoint builds, but without loading the BERT weights."""
    from colbert.infra.config import ColBERTConfig
    from colbert.modeling.tokenization import QueryTokenizer, DocTokenizer

    cfg = ColBERTConfig(
        query_maxlen=query_maxlen,
        doc_maxlen=doc_maxlen,
        dim=128,
        mask_punctuation=True,
        checkpoint=checkpoint,
    )
    return QueryTokenizer(cfg), DocTokenizer(cfg)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', required=True, choices=BEIR_DATASETS)
    ap.add_argument('--data_root', default='data')
    ap.add_argument('--output_dir', required=True)
    ap.add_argument('--batch_size', type=int, default=256)
    args = ap.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print('Loading tokenizers...')
    qt, dt = load_tokenizers()

    print(f'Loading {args.dataset}...')
    corpus, queries, _ = load_beir(args.data_root, args.dataset)

    # Same order and same tokenizer call as base.encode.encode_queries, so the
    # token ids line up with the qids stored in base_candidates.pt.
    qids = list(queries.keys())
    texts = [queries[q] for q in qids]
    chunks = []
    for i in range(0, len(texts), args.batch_size):
        ids, _ = qt.tensorize(texts[i:i + args.batch_size])
        chunks.append(ids.cpu())
    token_ids = torch.cat(chunks, dim=0) if chunks else torch.zeros(0, 32, dtype=torch.long)

    # cad.idf.compute_idf only needs the doc tokenizer.
    idf = compute_idf(SimpleNamespace(doc_tokenizer=dt), corpus, token_ids)

    torch.save(idf, out / 'idf.pt')
    torch.save({'qids': qids, 'token_ids': token_ids}, out / 'rerank_inputs.pt')
    print(f'Saved {out / "idf.pt"} ({len(idf)} tokens) and {out / "rerank_inputs.pt"} '
          f'({len(qids)} queries)')


if __name__ == '__main__':
    main()
