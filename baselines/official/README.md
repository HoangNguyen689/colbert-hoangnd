# Official ColBERTv2 baseline (BEIR)

Runs the **original ColBERTv2 retrieval pipeline** (official
[stanford-futuredata/ColBERT](https://github.com/stanford-futuredata/ColBERT) repo)
so we can compare against Mini-ColBERTv2 on the **same dataset with the same
settings**.

## Why this exists

The "Official ColBERTv2" row in the report must be measured, not guessed. The
numbers in the ColBERTv2 paper use `doc_maxlen=300` (and different
probe/candidate settings), so they are **not** comparable to the mini pipeline,
which uses `doc_maxlen=180`. These scripts pin the settings to match the mini
pipeline:

| Setting | Value |
|---|---|
| `query_maxlen` | 32 |
| `doc_maxlen` | 180 |
| `nbits` | 2 |
| top-k | 100 |

## Scripts

| File | Purpose |
|---|---|
| `prepare_beir.py` | Download a BEIR dataset and convert to ColBERT TSV format (`collection.tsv`, `queries.tsv`, `qrels.tsv`). |
| `run_official_colbertv2.py` | Build the residual index (nbits=2) and retrieve top-100. |
| `evaluate_official.py` | Compute nDCG@10 / MRR@10 / Recall@100; `--json-out` writes them for `run_evaluate.py`. |

## Usage (Colab GPU)

On Colab, run from inside a clone of the official ColBERT repo after
`pip install -e .` and `pip install faiss-gpu-cu12`:

```bash
# 1. Prepare the dataset in ColBERT TSV format
python prepare_beir.py --dataset nfcorpus --out data

# 2. Build index + retrieve (settings pinned to match mini)
python run_official_colbertv2.py --dataset nfcorpus --nbits 2 --k 100

# 3. Evaluate and save the metrics for the mini report table
python evaluate_official.py --dataset nfcorpus \
    --json-out /content/colbert-hoangnd/outputs/nfcorpus/official_baseline.json
```

Then, back in the mini repo:

```bash
python run_evaluate.py --dataset nfcorpus --output_dir outputs/nfcorpus --qrels outputs/nfcorpus/qrels_test.json
```

`run_evaluate.py` picks up `outputs/<dataset>/official_baseline.json` and prints
the `Official ColBERTv2` row automatically.

The whole loop is wrapped in [`Official_ColBERTv2_MultiBEIR.ipynb`](../../Official_ColBERTv2_MultiBEIR.ipynb).

## Notes

- The official repo needs a GPU for indexing; use a Colab T4.
- These scripts are intentionally standalone (no imports from `base/`) so they
  can be copied directly into the official ColBERT repo directory.
