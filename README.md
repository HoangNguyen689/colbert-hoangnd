# Mini-ColBERTv2

A minimal, educational re-implementation of the indexing and retrieval pipeline from **ColBERTv2** (checkpoint `colbert-ir/colbertv2.0`), evaluated on the SciFact dataset. It keeps the core algorithm — centroid-based quantization, 2-bit residual compression, IVF, and MaxSim — but strips away most of the engineering/optimization layers of the official repo so the code is easy to read, modify, and extend.

This repo also adds **CAD-MaxSim** (Context-Aware Diversified MaxSim) as a plug-in re-ranking layer.

---

## 1. How this differs from the official ColBERT repo

| Component | Official ColBERTv2 | Mini-ColBERTv2 (this repo) |
|---|---|---|
| Checkpoint | `colbert-ir/colbertv2.0` | **same** — no retraining |
| Encoder / token embeddings | BERT + projection + L2 normalize | **same**, calls `Checkpoint.queryFromText` / `docFromText` |
| K-means centroid building | sampling, FAISS, auto partition count, distributed support | simple `sklearn.KMeans`, fixed `K=8192` |
| Residual compression | `ResidualCodec`, learned bucket cutoffs/weights, 1/2-bit bit-pack | self-contained bucket quantization + optional bit-packing |
| Index format | many chunks, metadata, codec files, doclens | a few PyTorch tensors / Python dicts |
| IVF | optimized structure | `centroid_id → list[passage_ids]` |
| Candidate generation | optimized + tensorized | top centroids → union passage ids |
| Candidate pruning | multi-stage approximate pruning | optional top-N, otherwise decompress + MaxSim directly |
| Decompression | C++/CUDA optimized + CPU fallback | plain PyTorch / NumPy |
| MaxSim | optimized CPU/CUDA paths | clear PyTorch implementation |
| Multi-GPU / multiprocessing / resume / mmap | yes | removed for simplicity |
| Production speed | much faster | slower, but sufficient for SciFact |
| Readability | harder | very easy |
| CAD-MaxSim | not present | **added** |

In short:

> **Official repo** = ColBERTv2 algorithm + heavy engineering to scale and run fast.  
> **Mini-ColBERTv2** = keeps encoder + centroid + 2-bit residual + IVF + MaxSim, drops distributed/chunk/CUDA/multi-stage pruning so it can be understood and hacked.

We intentionally do **not** remove centroid/residual, because that would make it closer to ColBERT v1 than a faithful v2 implementation.

---

## 2. Repository structure

```
.
├── mini_colbert/
│   ├── encode.py           # checkpoint → Q/D embeddings
│   ├── build_index.py      # k-means → centroids + codes
│   ├── compress_residuals.py # 2-bit residual quantization
│   ├── residual.py         # quantize / dequantize helpers
│   ├── ivf.py              # centroid → passage ids
│   ├── search.py           # IVF candidate generation + MaxSim
│   ├── cad_maxsim.py       # CAD-MaxSim 4 ablations
│   ├── evaluate.py         # nDCG@10, MRR@10, Recall@100
│   ├── validate.py         # approximate vs exact MaxSim sanity check
│   └── compare_official.py # compare metrics with official baseline
├── Mini_ColBERTv2_SciFact.ipynb  # one-click Colab notebook
├── README.md
└── .gitignore
```

`data/` and `outputs/` are generated at runtime and are ignored by git.

---

## 3. Installation

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install torch transformers scikit-learn datasets beir tqdm numpy colbert-ai
```

> **Note:** `colbert-ai` is used only for `Checkpoint.queryFromText()` / `docFromText()` so we do not re-implement tokenizer/BERT. On Apple Silicon the official indexing path (FAISS/CUDA) is not used.

---

## 4. Running locally

### 4.1 Encode SciFact

```bash
python -m mini_colbert.encode \
  --data_root data \
  --output_dir outputs \
  --doc_batch_size 32 \
  --query_batch_size 32
```

This downloads SciFact via BEIR and produces:
- `outputs/doc_embeddings.pt`
- `outputs/query_embeddings.pt`
- `outputs/qrels_test.json`

### 4.2 Build index

```bash
python -m mini_colbert.build_index \
  --emb_path outputs/doc_embeddings.pt \
  --output_dir outputs \
  --K 8192 \
  --n_samples 100000

python -m mini_colbert.compress_residuals \
  --emb_path outputs/doc_embeddings.pt \
  --centroids_path outputs/centroids.pt \
  --codes_path outputs/codes.pt \
  --output_dir outputs \
  --nbits 2

python -m mini_colbert.ivf \
  --emb_path outputs/doc_embeddings.pt \
  --codes_path outputs/codes.pt \
  --output_dir outputs
```

### 4.3 Search

```bash
python -m mini_colbert.search \
  --index_dir outputs \
  --query_emb outputs/query_embeddings.pt \
  --output outputs/search_results.pt \
  --ncells 4 \
  --topk 100 \
  --device cpu
```

Use `--device cuda` on a GPU machine.

### 4.4 CAD-MaxSim ablations

```bash
python -m mini_colbert.cad_maxsim \
  --index_dir outputs \
  --query_emb outputs/query_embeddings.pt \
  --qrels outputs/qrels_test.json \
  --output_dir outputs \
  --ncells 4 \
  --topk 100 \
  --device cpu
```

This generates `outputs/rankings_{original,idf,adaptive,cad}.json`.

### 4.5 Evaluate

```bash
python -m mini_colbert.evaluate \
  --rankings outputs/rankings_original.json \
  --qrels outputs/qrels_test.json

python -m mini_colbert.compare_official \
  --rankings outputs/rankings_original.json \
  --qrels outputs/qrels_test.json
```

---

## 5. Running on Google Colab

Open [`Mini_ColBERTv2_SciFact.ipynb`](Mini_ColBERTv2_SciFact.ipynb) in Colab with a GPU runtime (T4 is enough). The notebook clones this repo, installs dependencies, runs the full pipeline, and prints metrics.

A GPU runtime is recommended for:
- full SciFact encoding
- `K=8192` k-means (still on CPU via sklearn, but faster end-to-end)
- 300-query evaluation

---

## 6. Results on SciFact (300 test queries)

Official ColBERTv2 baseline (`colbert-ir/colbertv2.0`):

| Metric | Official |
|---|---|
| nDCG@10 | 0.6438 |
| MRR@10 | 0.6120 |
| Recall@100 | 0.8842 |

Mini-ColBERTv2 (Mac M4, `K=8192`, sample 100k, `ncells=4`, 2-bit residual):

| Variant | nDCG@10 | MRR@10 | Recall@100 |
|---|---|---|---|
| Original MaxSim | 0.6308 | 0.5984 | 0.8676 |
| IDF (α=1, β=0) | 0.6166 | 0.5841 | 0.8659 |
| Adaptive (α=0, β=1) | 0.4863 | 0.4620 | 0.7712 |
| CAD (α=1, β=1) | 0.4693 | 0.4420 | 0.7746 |

The Original MaxSim result is within ~0.01–0.02 of the official baseline; the gap comes from our simplified quantization (uniform buckets, no learned cutoffs) and single-stage candidate pruning.

---

## 7. Validation

`validate.py` compares approximate search against exact MaxSim on the uncompressed embeddings for a few queries. Typical output:

```
Query: 0
Top-10 overlap: 8/10
Kendall tau on 2514 candidates: 0.7893
```

This confirms that decompression + MaxSim is implemented correctly; the remaining difference versus the official repo is mainly the simpler candidate-pruning pipeline.

---

## 8. Data

SciFact is downloaded automatically by `encode.py` using BEIR and stored under `data/`. Because the corpus, embeddings, and index files are large, they are excluded from git (see `.gitignore`). Users should run the pipeline themselves to generate them.

---

## 9. License

Code in this repo is provided for educational/research purposes. The ColBERT checkpoint is from [colbert-ir/colbertv2.0](https://huggingface.co/colbert-ir/colbertv2.0) and the SciFact dataset is from BEIR.
