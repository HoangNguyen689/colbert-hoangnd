# Mini-ColBERTv2

Phiên bản tái cài đặt tối giản, dễ đọc của pipeline indexing và retrieval từ **ColBERTv2** (checkpoint `colbert-ir/colbertv2.0`), đánh giá trên tập SciFact. Repo giữ lại thuật toán cốt lõi — centroid quantization, nén residual 2-bit, IVF và MaxSim — nhưng lược bỏ phần lớn lớp engineering/optimization của repo chính thức để dễ hiểu, dễ sửa và dễ mở rộng.

Repo cũng bổ sung thêm lớp **CAD-MaxSim** (Context-Aware Diversified MaxSim) như một bước re-rank đơn giản.

---

## 1. Khác biệt so với repo ColBERT chính thức

| Thành phần | Official ColBERTv2 | Mini-ColBERTv2 (repo này) |
|---|---|---|
| Checkpoint | `colbert-ir/colbertv2.0` | **giống** — không train lại |
| Encoder / token embeddings | BERT + projection + L2 normalize | **giống**, gọi `Checkpoint.queryFromText` / `docFromText` |
| Xây dựng centroid | sampling, FAISS, tự chọn số partitions, hỗ trợ distributed | `sklearn.KMeans` đơn giản, cố định `K=8192` |
| Nén residual | `ResidualCodec`, học bucket cutoffs/weights, bit-pack 1/2-bit | tự viết bucket quantization + bit-pack tùy chọn |
| Định dạng index | nhiều chunk, metadata, codec files, doclens | vài tensor/file Python đơn giản |
| IVF | cấu trúc tối ưu | `centroid_id → list[passage_ids]` |
| Sinh candidate | tối ưu + tensorized | top centroids → union passage ids |
| Candidate pruning | nhiều tầng approximate pruning | optional top-N, hoặc decompress + MaxSim luôn |
| Giải nén | C++/CUDA optimized + CPU fallback | PyTorch / NumPy |
| MaxSim | CPU/CUDA optimized | PyTorch rõ ràng |
| Multi-GPU / multiprocessing / resume / mmap | có | bỏ |
| Tốc độ production | nhanh hơn nhiều | chậm hơn nhưng đủ cho SciFact |
| Dễ đọc | khá khó | rất dễ |
| CAD-MaxSim | không có | **có** |

Tóm lại:

> **Official repo** = thuật toán ColBERTv2 + engineering nặng để scale và chạy nhanh.  
> **Mini-ColBERTv2** = giữ encoder + centroid + residual 2-bit + IVF + MaxSim, bỏ distributed/chunk/CUDA/pruning phức tạp để dễ hiểu và sửa được.

Chúng tôi cố tình **không** bỏ centroid/residual, vì nếu bỏ thì sẽ gần ColBERT v1 hơn là v2.

---

## 2. Cấu trúc repo

```
.
├── mini_colbert/
│   ├── encode.py              # checkpoint → Q/D embeddings
│   ├── build_index.py         # k-means → centroids + codes
│   ├── compress_residuals.py  # lượng tử hóa residual 2-bit
│   ├── residual.py            # helper quantize / dequantize
│   ├── ivf.py                 # centroid → passage ids
│   ├── search.py              # IVF candidate generation + MaxSim
│   ├── cad_maxsim.py          # 4 ablations của CAD-MaxSim
│   ├── evaluate.py            # nDCG@10, MRR@10, Recall@100
│   ├── validate.py            # kiểm tra approximate vs exact MaxSim
│   └── compare_official.py    # so sánh với baseline official
├── Mini_ColBERTv2_SciFact.ipynb  # notebook chạy trên Colab
├── README.md
└── .gitignore
```

`data/` và `outputs/` được sinh ra khi chạy và đã bị git ignore.

---

## 3. Cài đặt

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install torch transformers scikit-learn datasets beir tqdm numpy colbert-ai
```

> **Lưu ý:** `colbert-ai` chỉ dùng để gọi `Checkpoint.queryFromText()` / `docFromText()`, tránh phải tự implement tokenizer/BERT. Trên Apple Silicon, phần indexing chính thức (FAISS/CUDA) không được dùng.

---

## 4. Chạy local

### 4.1 Encode SciFact

```bash
python -m mini_colbert.encode \
  --data_root data \
  --output_dir outputs \
  --doc_batch_size 32 \
  --query_batch_size 32
```

Lệnh này tải SciFact qua BEIR và sinh ra:
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

Trên máy có GPU thì dùng `--device cuda`.

### 4.4 CAD-MaxSim 4 ablations

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

Sinh ra `outputs/rankings_{original,idf,adaptive,cad}.json`.

### 4.5 Đánh giá

```bash
python -m mini_colbert.evaluate \
  --rankings outputs/rankings_original.json \
  --qrels outputs/qrels_test.json

python -m mini_colbert.compare_official \
  --rankings outputs/rankings_original.json \
  --qrels outputs/qrels_test.json
```

---

## 5. Chạy trên Google Colab

Mở [`Mini_ColBERTv2_SciFact.ipynb`](Mini_ColBERTv2_SciFact.ipynb) trong Colab với runtime GPU (T4 là đủ). Notebook tự clone repo, cài dependency, chạy full pipeline và in metrics.

Nên dùng GPU runtime cho:
- encode toàn bộ SciFact
- k-means `K=8192` (vẫn chạy CPU qua sklearn nhưng end-to-end nhanh hơn)
- đánh giá 300 query

---

## 6. Kết quả trên SciFact (300 test queries)

Baseline ColBERTv2 chính thức (`colbert-ir/colbertv2.0`):

| Metric | Official |
|---|---|
| nDCG@10 | 0.6438 |
| MRR@10 | 0.6120 |
| Recall@100 | 0.8842 |

Mini-ColBERTv2 (Mac M4, `K=8192`, sample 100k, `ncells=4`, residual 2-bit):

| Biến thể | nDCG@10 | MRR@10 | Recall@100 |
|---|---|---|---|
| Original MaxSim | 0.6308 | 0.5984 | 0.8676 |
| IDF (α=1, β=0) | 0.6166 | 0.5841 | 0.8659 |
| Adaptive (α=0, β=1) | 0.4863 | 0.4620 | 0.7712 |
| CAD (α=1, β=1) | 0.4693 | 0.4420 | 0.7746 |

Original MaxSim chỉ thua official ~0.01–0.02; khoảng cách đến từ lượng tử hóa đơn giản (uniform buckets, chưa học cutoffs) và chỉ dùng một tầng candidate pruning.

---

## 7. Kiểm tra độ đúng

`validate.py` so sánh approximate search với exact MaxSim trên embedding gốc (chưa nén) cho vài query. Kết quả điển hình:

```
Query: 0
Top-10 overlap: 8/10
Kendall tau on 2514 candidates: 0.7893
```

Điều này xác nhận decompress + MaxSim implement đúng; phần khác biệt còn lại so với official chủ yếu là pipeline pruning đơn giản hơn.

---

## 8. Dữ liệu

SciFact được tải tự động bởi `encode.py` qua BEIR và lưu trong `data/`. Vì corpus, embeddings và index khá nặng, chúng bị loại khỏi git (xem `.gitignore`). Ngườii dùng cần tự chạy pipeline để sinh ra.

---

## 9. License

Code trong repo này cung cấp cho mục đích giáo dục/nghiên cứu. Checkpoint ColBERT từ [colbert-ir/colbertv2.0](https://huggingface.co/colbert-ir/colbertv2.0) và tập SciFact từ BEIR.
