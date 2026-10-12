# Mini-ColBERTv2

Phiên bản cài đặt tối giản của pipeline indexing và retrieval từ **ColBERTv2** (checkpoint `colbert-ir/colbertv2.0`), đánh giá trên tập SciFact. Repo giữ lại thuật toán cốt lõi - centroid quantization, nén residual 2-bit, IVF và MaxSim - nhưng lược bỏ phần lớn lớp engineering/optimization của repo chính thức để dễ hiểu, dễ sửa và dễ mở rộng.

Repo cũng bổ sung thêm lớp **CAD-MaxSim** (Candidate-Adaptive Discriminative MaxSim) như một bước re-rank đơn giản.

---

## 1. Khác biệt so với repo ColBERT chính thức

| Thành phần | Official ColBERTv2 | Mini-ColBERTv2 (repo này) |
|---|---|---|
| Checkpoint | `colbert-ir/colbertv2.0` | **giống** - không train lại |
| Encoder / token embeddings | BERT + projection + L2 normalize | **giống**, gọi `Checkpoint.queryFromText` / `docFromText` |
| Xây dựng centroid | sampling, FAISS, tự chọn số partitions, hỗ trợ distributed | `sklearn.MiniBatchKMeans` + GPU assignment, cố định `K=8192` |
| Nén residual | `ResidualCodec`, học bucket cutoffs/weights, bit-pack 1/2-bit | quantile buckets + reconstruction weights + bit-pack tùy chọn |
| Định dạng index | nhiều chunk, metadata, codec files, doclens | vài tensor/file Python đơn giản |
| IVF | cấu trúc tối ưu | `centroid_id → list[passage_ids]` |
| Sinh candidate | tối ưu + tensorized | top centroids → union passage ids |
| Candidate pruning | nhiều tầng approximate pruning | optional top-N, hoặc decompress + MaxSim luôn |
| Giải nén | C++/CUDA optimized + CPU fallback | PyTorch / NumPy |
| MaxSim | CPU/CUDA optimized | PyTorch rõ ràng |
| Multi-GPU / multiprocessing / resume / mmap | có | bỏ |
| Tốc độ production | nhanh hơn nhiều | chậm hơn nhưng đủ cho SciFact |
| CAD-MaxSim | không có | **có** |

Tóm lại:

> **Official repo** = thuật toán ColBERTv2 + heavy engineering để scale và chạy nhanh.  
> **Mini-ColBERTv2** = giữ encoder + centroid + residual 2-bit + IVF + MaxSim, bỏ distributed/chunk/CUDA/pruning phức tạp để dễ hiểu và sửa được.

---

## 2. Cấu trúc repo

```
.
├── base/                      # Mini-ColBERTv2 thuần
│   ├── encode.py              # checkpoint → Q/D embeddings
│   ├── index.py               # k-means → centroids + codes
│   ├── compress_residuals.py  # lượng tử hóa residual 2-bit
│   ├── residual.py            # helper quantize / dequantize
│   ├── ivf.py                 # centroid → passage ids
│   ├── search.py              # IVF candidate generation + MaxSim
│   └── maxsim.py              # per-token MaxSim + score
├── cad/                       # CAD-MaxSim extension
│   ├── idf.py                 # corpus-level IDF
│   ├── weighting.py           # CAD weight formula
│   └── rerank.py              # rerank từ base candidates
├── evaluation/                # chỉ đọc ranking → metrics
│   ├── metrics.py             # nDCG@10, MRR@10, Recall@100
│   └── compare_official.py    # so sánh với baseline official
├── run_base.py                # pipeline BASE end-to-end
├── run_cad.py                 # pipeline CAD end-to-end
├── run_evaluate.py            # in bảng metrics
├── Mini_ColBERTv2_SciFact.ipynb  # notebook chạy trên Colab
├── README.md
└── .gitignore
```

`data/` và `outputs/` được sinh ra khi chạy.

**Nguyên tắc tách biệt:**
- `base/` không biết gì về CAD.
- `cad/` chỉ đọc `outputs/base_candidates.pt` (chứa `pids`, `scores`, `token_scores`); không chạy retrieval hay giải nén lại.
- `evaluation/` chỉ đọc ranking JSON + qrels; không cần checkpoint/GPU/embeddings.

---

## 3. Cài đặt

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install torch transformers scikit-learn datasets beir tqdm numpy colbert-ai
```

> **Lưu ý:** `colbert-ai` chỉ dùng để gọi `Checkpoint.queryFromText()` / `docFromText()`, tránh phải tự implement tokenizer/BERT. Trên Apple Silicon, phần indexing chính thức (FAISS/CUDA) không được dùng. Nếu gặp lỗi `all_tied_weights_keys` với `transformers 5.x`, hãy pin `transformers<5.0` (ví dụ `transformers==4.57.6`).

---

## 4. Chạy local

### 4.1 Chạy toàn bộ BASE

```bash
python run_base.py \
  --data_root data \
  --output_dir outputs \
  --K 8192 \
  --n_samples 100000 \
  --ncells 4 \
  --candidate_topn 200 \
  --topk 100 \
  --device cpu \
  --doc_batch_size 32 \
  --query_batch_size 32
```

Lệnh này tải SciFact qua BEIR, encode, build index, nén residual, build IVF, search, và sinh ra:
- `outputs/base_candidates.pt` — candidate set + per-token MaxSim `c_i(d)`
  - `candidate_pids`: list length `candidate_topn` (mặc định 200)
  - `candidate_scores`: list length `candidate_topn`
  - `token_scores`: tensor shape `(candidate_topn, 32)`
- `outputs/rankings_original.json` — ranking Original MaxSim top-`output_topk` (mặc định 100)

Trên máy có GPU thì dùng `--device cuda`.

### 4.2 Chạy CAD

```bash
python run_cad.py --output_dir outputs --topk 100
```

CAD **không chạy retrieval lại**, chỉ đọc `base_candidates.pt`, tính IDF/std-dev, rồi rerank cùng candidate set với 4 ablations:
- `outputs/rankings_original.json`
- `outputs/rankings_idf.json`
- `outputs/rankings_adaptive.json`
- `outputs/rankings_cad.json`

### 4.3 Đánh giá

```bash
python run_evaluate.py --output_dir outputs --qrels outputs/qrels_test.json
```

In bảng so sánh Official / Mini Base / IDF / Adaptive / CAD.

---

## 5. Chạy trên Google Colab

Mở [`Mini_ColBERTv2_SciFact.ipynb`](Mini_ColBERTv2_SciFact.ipynb) trong Colab với runtime GPU (T4 là đủ). Notebook tự clone repo, cài dependency, rồi chạy 3 lệnh:

```bash
python run_base.py --output_dir outputs --K 8192 --n_samples 200000 --ncells 4 --candidate_topn 200 --topk 100 --device cuda
python run_cad.py --output_dir outputs --topk 100
python run_evaluate.py --output_dir outputs
```

Nên dùng GPU runtime cho encode full SciFact và search; CAD và evaluation chạy CPU cũng nhanh.

---

## 6. Kết quả trên SciFact (300 test queries)

Chạy trên Google Colab T4 với checkpoint `colbert-ir/colbertv2.0`, `K=8192`, `n_samples=200000`, `ncells=4`, `candidate_topn=200`, `output_topk=100`, residual 2-bit.

| Biến thể | nDCG@10 | MRR@10 | Recall@100 |
|---|---|---|---|---|
| Official ColBERTv2 | 0.6438 | 0.6120 | 0.8842 |
| Mini Base (2-bit IVF) | 0.6411 | 0.6103 | 0.8916 |
| IDF (α=1, β=0) | **0.6656** | **0.6367** | 0.8993 |
| Adaptive (α=0, β=1) | 0.6210 | 0.5903 | 0.8953 |
| CAD (α=1, β=1) | 0.6495 | 0.6204 | 0.8987 |

Nhận xét:

- Mini Base gần sát Official ColBERTv2 → centroid + residual 2-bit + IVF được implement đúng.
- IDF đơn thuần cho kết quả tốt nhất (+2.45 nDCG@10, +2.64 MRR@10 so với Mini Base).
- Adaptive (std-dev theo candidate pool) làm giảm chất lượng, có thể vì 200 candidates quá ít để ước lượng độ quan trọng token ổn định.
- CAD kết hợp IDF + Adaptive thua IDF đơn thuần, cho thấy thành phần β cần được tinh chỉnh (grid-search) hoặc thay thế bằng độ đo khác.

---

## 7. Dữ liệu

SciFact được tải tự động bởi `encode.py` qua BEIR và lưu trong `data/`. Vì corpus, embeddings và index khá nặng, chúng bị loại khỏi git (xem `.gitignore`). Ngườii dùng cần tự chạy pipeline để sinh ra.

---

## 8. License

Code trong repo này cung cấp cho mục đích học tập - nghiên cứu. Checkpoint ColBERT từ [colbert-ir/colbertv2.0](https://huggingface.co/colbert-ir/colbertv2.0) và tập SciFact từ BEIR.
