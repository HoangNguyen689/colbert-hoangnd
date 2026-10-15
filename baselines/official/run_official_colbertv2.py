"""
baselines/official/run_official_colbertv2.py
Run the ORIGINAL ColBERTv2 retrieval pipeline (official repo) on a prepared
BEIR dataset.

Settings are pinned to match the mini pipeline for a fair comparison:
  query_maxlen=32, doc_maxlen=180, nbits=2, top-k=100.

Standalone on purpose: copy this file into the official ColBERT repo working
directory on Colab (it imports colbert.* directly).
"""
from pathlib import Path
import argparse
import time

from colbert import Indexer, Searcher
from colbert.data import Queries
from colbert.infra import Run, RunConfig, ColBERTConfig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="scifact")
    ap.add_argument("--data", default=None, help="default: data/<dataset>_colbert")
    ap.add_argument("--root", default="experiments")
    ap.add_argument("--experiment", default=None, help="default: <dataset>-original")
    ap.add_argument("--index-name", default=None, help="default: <dataset>.nbits<nbits>")
    ap.add_argument("--checkpoint", default="colbert-ir/colbertv2.0")
    ap.add_argument("--nbits", type=int, default=2)
    ap.add_argument("--k", type=int, default=100)
    ap.add_argument("--doc-maxlen", type=int, default=180)
    ap.add_argument("--query-maxlen", type=int, default=32)
    ap.add_argument("--ranking-out", default=None, help="default: ranking.<dataset>.tsv")
    ap.add_argument("--skip-index", action="store_true")
    args = ap.parse_args()

    ds = args.dataset
    data = Path(args.data or f"data/{ds}_colbert").resolve()
    root = str(Path(args.root).resolve())
    experiment = args.experiment or f"{ds}-original"
    index_name = args.index_name or f"{ds}.nbits{args.nbits}"
    collection = str(data / "collection.tsv")
    queries_path = str(data / "queries.tsv")
    output = str((Path.cwd() / (args.ranking_out or f"ranking.{ds}.tsv")).resolve())

    with Run().context(RunConfig(nranks=1, experiment=experiment)):
        if not args.skip_index:
            cfg = ColBERTConfig(
                nbits=args.nbits,
                root=root,
                doc_maxlen=args.doc_maxlen,
                query_maxlen=args.query_maxlen,
            )
            t0 = time.time()
            indexer = Indexer(checkpoint=args.checkpoint, config=cfg)
            indexer.index(name=index_name, collection=collection, overwrite=True)
            print(f"INDEX_SECONDS={time.time() - t0:.2f}")

        search_cfg = ColBERTConfig(root=root, query_maxlen=args.query_maxlen)
        searcher = Searcher(index=index_name, collection=collection, config=search_cfg)
        queries = Queries(queries_path)

        t0 = time.time()
        ranking = searcher.search_all(queries, k=args.k)
        print(f"SEARCH_SECONDS={time.time() - t0:.2f}")

        # ColBERT's Ranking.save() refuses to overwrite an existing file, so clear
        # any leftovers from a previous run first.
        out_path = Path(output)
        for leftover in (out_path, Path(str(out_path) + ".meta")):
            if leftover.exists():
                leftover.unlink()

        saved = ranking.save(output)
        print("RANKING=", saved)


if __name__ == "__main__":
    main()
