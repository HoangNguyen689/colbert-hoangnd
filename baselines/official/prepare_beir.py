"""
baselines/official/prepare_beir.py
Download a BEIR dataset and convert it into ColBERT's TSV format.

Produces, under <out>/<dataset>_colbert/:
  collection.tsv   -> pid<TAB>text<TAB>title   (pid = zero-based line index)
  queries.tsv      -> qid<TAB>text
  qrels.tsv        -> qid<TAB>pid<TAB>rel
  docid_map.json   -> int pid -> original BEIR corpus id
  qid_map.json     -> int qid -> original BEIR query id

Standalone on purpose: this file is meant to be copied into the official
ColBERT repo working directory on Colab (so it must not import from base/).
"""
from pathlib import Path
import argparse
import csv
import json
import urllib.request
import zipfile

BEIR_URL = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip"


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def clean(text):
    return str(text).replace("\t", " ").replace("\n", " ").strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="scifact")
    ap.add_argument("--out", default="data")
    args = ap.parse_args()

    archive = subdir = args.dataset
    out = Path(args.out)
    raw = out / "raw"
    cooked = out / f"{args.dataset}_colbert"
    raw.mkdir(parents=True, exist_ok=True)
    cooked.mkdir(parents=True, exist_ok=True)

    zip_path = raw / f"{archive}.zip"
    if not zip_path.exists():
        print(f"Downloading {archive}...")
        urllib.request.urlretrieve(BEIR_URL.format(archive), zip_path)

    extracted = raw / archive
    if not extracted.exists():
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(raw)

    src = raw / subdir
    corpus_path = src / "corpus.jsonl"
    queries_path = src / "queries.jsonl"
    qrels_path = src / "qrels" / "test.tsv"
    assert corpus_path.exists(), corpus_path
    assert queries_path.exists(), queries_path
    assert qrels_path.exists(), qrels_path

    docs = list(read_jsonl(corpus_path))
    queries = list(read_jsonl(queries_path))

    # ColBERT's TSV collection loader requires pid == zero-based line index.
    doc_orig_to_int = {str(d["_id"]): i for i, d in enumerate(docs)}
    q_orig_to_int = {str(q["_id"]): i for i, q in enumerate(queries)}

    with open(cooked / "collection.tsv", "w", encoding="utf-8") as f:
        for pid, d in enumerate(docs):
            text = clean(d.get("text", ""))
            title = clean(d.get("title", ""))
            # ColBERT supports pid<TAB>passage<TAB>title.
            f.write(f"{pid}\t{text}\t{title}\n")

    with open(cooked / "queries.tsv", "w", encoding="utf-8") as f:
        for qid, q in enumerate(queries):
            f.write(f"{qid}\t{clean(q.get('text', ''))}\n")

    mapped_qrels = []
    with open(qrels_path, encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            oq = str(row["query-id"])
            od = str(row["corpus-id"])
            rel = int(row["score"])
            if oq in q_orig_to_int and od in doc_orig_to_int:
                mapped_qrels.append((q_orig_to_int[oq], doc_orig_to_int[od], rel))

    with open(cooked / "qrels.tsv", "w", encoding="utf-8") as f:
        f.write("qid\tpid\trel\n")
        for qid, pid, rel in mapped_qrels:
            f.write(f"{qid}\t{pid}\t{rel}\n")

    with open(cooked / "docid_map.json", "w", encoding="utf-8") as f:
        json.dump({str(v): k for k, v in doc_orig_to_int.items()}, f, indent=2)
    with open(cooked / "qid_map.json", "w", encoding="utf-8") as f:
        json.dump({str(v): k for k, v in q_orig_to_int.items()}, f, indent=2)

    print(f"Prepared {len(docs)} documents, {len(queries)} queries, {len(mapped_qrels)} qrels")
    print("Output:", cooked.resolve())


if __name__ == "__main__":
    main()
