# Mini-ColBERTv2

Phiên bản cài đặt tối giản của pipeline indexing và retrieval từ **ColBERTv2** (checkpoint `colbert-ir/colbertv2.0`), đánh giá trên tập SciFact. Repo giữ lại thuật toán cốt lõi - centroid quantization, nén residual 2-bit, IVF và MaxSim - nhưng lược bỏ phần lớn lớp engineering/optimization của repo chính thức để dễ hiểu, dễ sửa và dễ mở rộng.

Repo cũng bổ sung thêm lớp **CAD-MaxSim** (Context-Aware Diversified MaxSim) như một bước re-rank đơn giản.

---

## 1. Khác biệt so với repo ColBERT chính thức

| Thành phần | Official ColBERTv2 | Mini-ColBERTv2 (repo này) |
|---|---|---|
| Checkpoint | `colbert-ir/colbertv2.0` | **giống** - không train lại |
| Encoder / token embeddings | BERT + projection + L2 normalize | **giống**, gọi `Checkpoint.queryFromText` / `docFromText` |
| Xây dựng centroid | sampling, FAISS, tự chọn số partitions, hỗ trợ distributed | `sklearn.KMeans` đơn giản, cố định `K=8192` |
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
  --topk 100 \
  --device cpu \
  --doc_batch_size 32 \
  --query_batch_size 32
```

Lệnh này tải SciFact qua BEIR, encode, build index, nén residual, build IVF, search, và sinh ra:
- `outputs/base_candidates.pt` — candidate set + per-token MaxSim `c_i(d)`
- `outputs/rankings_original.json` — ranking Original MaxSim

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
python run_base.py --output_dir outputs --K 8192 --n_samples 200000 --ncells 4 --topk 100 --device cuda
python run_cad.py --output_dir outputs --topk 100
python run_evaluate.py --output_dir outputs
```

Nên dùng GPU runtime cho encode full SciFact và search; CAD và evaluation chạy CPU cũng nhanh.

---

## 6. Kết quả trên SciFact (300 test queries)

Kết quả được chạy trực tiếp từ commit hiện tại trên Mac M4 (`K=8192`, sample 100k, `ncells=4`, residual 2-bit, top-100 rerank).

Baseline ColBERTv2 chính thức (`colbert-ir/colbertv2.0`):

| Metric | Official |
|---|---|
| nDCG@10 | 0.6438 |
| MRR@10 | 0.6120 |
| Recall@100 | 0.8842 |

Mini-ColBERTv2:

| Biến thể | nDCG@10 | MRR@10 | Recall@100 |
|---|---|---|---|
| Exact MaxSim (không nén) | 0.6464 | 0.6172 | 0.8942 |
| Original MaxSim (2-bit IVF) | 0.6466 | 0.6147 | 0.8942 |
| IDF (α=1, β=0) | **0.6731** | **0.6440** | 0.8942 |
| Adaptive (α=0, β=1) | 0.6278 | 0.5961 | 0.8942 |
| CAD (α=1, β=1) | 0.6552 | 0.6264 | 0.8942 |

Original MaxSim đã sát với official (thậm chí hơn một chút về nDCG/Recall) và rất gần exact MaxSim không nén, chứng tỏ centroid + residual 2-bit + IVF được implement đúng. IDF weighting cho kết quả tốt nhất trong 4 ablations.

---

## 7. Kiểm tra độ đúng

`validate.py` so sánh approximate search với exact MaxSim trên embedding gốc (chưa nén) cho vài query. Kết quả điển hình sau khi sửa centroid normalize + quantile residual:

```
Query: 0
Top-10 overlap: 8/10
Kendall tau on 2388 candidates: 0.8809

Query: 2
Top-10 overlap: 9/10
Kendall tau on 1768 candidates: 0.9103

Query: 4
Top-10 overlap: 9/10
Kendall tau on 2170 candidates: 0.9227
```

Điều này xác nhận decompress + MaxSim implement đúng; khoảng cách còn lại với exact MaxSim chủ yếu là do lượng tử hóa residual.

---

## 8. Dữ liệu

SciFact được tải tự động bởi `encode.py` qua BEIR và lưu trong `data/`. Vì corpus, embeddings và index khá nặng, chúng bị loại khỏi git (xem `.gitignore`). Ngườii dùng cần tự chạy pipeline để sinh ra.

---

## 9. License

Code trong repo này cung cấp cho mục đích học tập - nghiên cứu. Checkpoint ColBERT từ [colbert-ir/colbertv2.0](https://huggingface.co/colbert-ir/colbertv2.0) và tập SciFact từ BEIR.
