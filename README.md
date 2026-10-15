# Mini-ColBERTv2

Phiên bản cài đặt tối giản của pipeline indexing và retrieval từ **ColBERTv2** (checkpoint `colbert-ir/colbertv2.0`), đánh giá trên nhiều tập BEIR (SciFact, NFCorpus, SCIDOCS, FiQA, ArguAna). Repo giữ lại thuật toán cốt lõi - centroid quantization, nén residual 2-bit, IVF và MaxSim - nhưng lược bỏ phần lớn lớp engineering/optimization của repo chính thức để dễ hiểu, dễ sửa và dễ mở rộng.

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
│   ├── datasets.py            # danh sách BEIR datasets (dependency-free)
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
│   ├── rerank.py              # rerank từ base candidates
│   └── tune_cad.py            # grid-search alpha/beta (hiện cố định α=1.0)
├── evaluation/                # chỉ đọc ranking → metrics
│   ├── metrics.py             # nDCG@10, MRR@10, Recall@100
│   └── compare_official.py    # so sánh với baseline official
├── baselines/
│   └── official/              # chạy ColBERTv2 gốc cùng settings để làm baseline
│       ├── prepare_beir.py
│       ├── run_official_colbertv2.py
│       └── evaluate_official.py
├── run_base.py                # pipeline BASE end-to-end
├── run_cad.py                 # pipeline CAD end-to-end
├── run_evaluate.py            # in bảng metrics (kèm hàng Official nếu có)
├── Mini_ColBERTv2_SciFact.ipynb      # notebook chạy 1 dataset trên Colab
├── Mini_ColBERTv2_MultiBEIR.ipynb    # notebook chạy nhiều dataset (base+cad+eval)
├── Official_ColBERTv2_MultiBEIR.ipynb # notebook chạy official baseline
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
  --dataset scifact \
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

`--dataset` hỗ trợ các tập BEIR: `scifact` (mặc định), `nfcorpus`, `scidocs`, `fiqa`, `trec-covid`, `arguana`. Dataset được tải tự động về `data/<dataset>/` khi chạy lần đầu.

Lệnh này tải SciFact qua BEIR, encode, build index, nén residual, build IVF, search, và sinh ra:
- `outputs/base_candidates.pt` — candidate set + per-token MaxSim `c_i(d)`
  - `candidate_pids`: list length `candidate_topn` (mặc định 200)
  - `candidate_scores`: list length `candidate_topn`
  - `token_scores`: tensor shape `(candidate_topn, 32)`
- `outputs/rankings_original.json` — ranking Original MaxSim top-`output_topk` (mặc định 100)

Trên máy có GPU thì dùng `--device cuda`.

### 4.2 Chạy CAD

```bash
python run_cad.py --dataset scifact --output_dir outputs --topk 100
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

In bảng so sánh Official / Mini Base / IDF / Adaptive / CAD. Hàng `Official` chỉ hiện khi có baseline cho dataset (đọc từ `outputs/<dataset>/official_baseline.json`, hoặc bảng dựng sẵn cho SciFact).

---

## 5. Chạy trên Google Colab

Ba notebook (runtime GPU T4):

1. [`Mini_ColBERTv2_SciFact.ipynb`](Mini_ColBERTv2_SciFact.ipynb) — chạy 1 dataset (SciFact).
2. [`Mini_ColBERTv2_MultiBEIR.ipynb`](Mini_ColBERTv2_MultiBEIR.ipynb) — chạy nhiều dataset (`nfcorpus`, `scidocs`, `fiqa`, `arguana`), mỗi dataset vào `outputs/<dataset>/`.
3. [`Official_ColBERTv2_MultiBEIR.ipynb`](Official_ColBERTv2_MultiBEIR.ipynb) — chạy ColBERTv2 gốc cùng settings để sinh `official_baseline.json`.

Nên dùng GPU runtime cho encode và search; CAD và evaluation chạy CPU cũng nhanh.

> `trec-covid` (171k docs, ~14GB) mặc định **tắt** để tránh OOM trên T4 12GB. Muốn chạy thì bỏ comment dòng `# DATASETS += OPTIONAL_DATASETS` trong cell cấu hình của notebook (nên dùng high-RAM).

**Chia sẻ giữa các phiên Colab (quan trọng):** mỗi notebook Colab có runtime riêng và `/content` mất khi hết phiên. Hai notebook multi-BEIR mount Google Drive và chỉ đồng bộ **các file kết quả nhỏ** vào `MyDrive/colbert-run/<dataset>/`:

- `rankings_{original,idf,adaptive,cad}.json`, `qrels_test.json` (từ notebook mini)
- `base_candidates.pt` (để rerank/tune CAD mà không cần encode lại)
- `official_baseline.json` (từ notebook official)

Nhờ vậy mở phiên mới không phải chạy lại encode/index (~20–25 phút cho 4 tập). Các file nặng (embeddings, residual, IVF) được **giữ local** vì nặng và tái tạo được; đừng đưa lên Drive (dễ tràn 15 GB và đọc chậm hơn).

---

## 6. So sánh với Official ColBERTv2

Để so sánh công bằng, official ColBERTv2 phải chạy **cùng settings** với bản mini: `query_maxlen=32`, `doc_maxlen=180`, `nbits=2`, top-100. Số trong paper dùng `doc_maxlen=300` nên không so trực tiếp được.

Chạy `Official_ColBERTv2_MultiBEIR.ipynb` (Colab GPU); notebook dùng `baselines/official/` để:

1. tải dataset và convert sang ColBERT TSV;
2. build index residual (`nbits=2`) + retrieve top-100;
3. tính metrics và ghi `outputs/<dataset>/official_baseline.json`.

Sau đó `run_evaluate.py` tự hiện hàng `Official ColBERTv2` cho từng dataset.

---

## 7. Kết quả đo trên Colab T4

Chạy trên Google Colab (GPU T4) với checkpoint `colbert-ir/colbertv2.0`, `K=8192`, `n_samples=200000`, `ncells=4`, `candidate_topn=200`, `output_topk=100`, residual 2-bit, `query_maxlen=32`, `doc_maxlen=180`.

Hàng `Official ColBERTv2` được đo bằng `baselines/official/` với **cùng settings** (không lấy số paper dùng `doc_maxlen=300`).

Số query test: SciFact 300, NFCorpus 323, SCIDOCS 1.000, FiQA 648, ArguAna 1.406. `trec-covid` **không chạy**: 171k docs, `doc_embeddings.pt` ~14 GB → OOM trên T4 12.7 GB RAM.

### 7.1 nDCG@10

| Dataset | Official | Mini Base | IDF (α=1, β=0) | Adaptive (α=0, β=1) | CAD (α=1, β=1) |
|---|---|---|---|---|---|
| SciFact | 0.6438 | 0.6411 | **0.6656** | 0.6210 | 0.6495 |
| NFCorpus | 0.3299 | **0.3289** | 0.3285 | 0.3224 | 0.3250 |
| SCIDOCS | 0.1549 | 0.1563 | **0.1619** | 0.1439 | 0.1551 |
| FiQA | 0.3410 | 0.3401 | **0.3414** | 0.3209 | 0.3214 |
| ArguAna | 0.3366 | 0.3331 | 0.3346 | 0.3315 | **0.3352** |

### 7.2 MRR@10

| Dataset | Official | Mini Base | IDF | Adaptive | CAD |
|---|---|---|---|---|---|
| SciFact | 0.6120 | 0.6103 | **0.6367** | 0.5903 | 0.6204 |
| NFCorpus | 0.5491 | **0.5487** | 0.5459 | 0.5342 | 0.5413 |
| SCIDOCS | 0.2744 | 0.2791 | **0.2900** | 0.2640 | 0.2810 |
| FiQA | 0.4161 | 0.4152 | **0.4167** | 0.3892 | 0.3959 |
| ArguAna | 0.2245 | 0.2211 | **0.2255** | 0.2222 | 0.2248 |

### 7.3 Recall@100

| Dataset | Official | Mini Base | IDF | Adaptive | CAD |
|---|---|---|---|---|---|
| SciFact | 0.8842 | 0.8916 | **0.8993** | 0.8953 | 0.8987 |
| NFCorpus | **0.2738** | 0.2722 | 0.2693 | 0.2704 | 0.2675 |
| SCIDOCS | 0.3555 | 0.3590 | **0.3614** | 0.3473 | 0.3555 |
| FiQA | 0.6249 | 0.6312 | **0.6346** | 0.6218 | 0.6226 |
| ArguAna | 0.9201 | 0.9090 | 0.9111 | 0.9189 | **0.9203** |

### 7.4 Thay đổi nDCG@10 so với Mini Base (%)

| Dataset | IDF | Adaptive | CAD |
|---|---:|---:|---:|
| SciFact | **+3.8** | −3.1 | +1.3 |
| NFCorpus | −0.1 | −2.0 | −1.2 |
| SCIDOCS | **+3.6** | −7.9 | −0.8 |
| FiQA | +0.4 | −5.6 | −5.5 |
| ArguAna | +0.5 | −0.5 | +0.6 |

### 7.5 Nhận xét

- **Mini Base bám sát Official trên cả 5 tập** (lệch ≤ 1%) → centroid quantization + residual 2-bit + IVF + MaxSim implement đúng, và so sánh là công bằng (cùng settings).
- **IDF là thành phần duy nhất đóng góp ổn định**: cải thiện rõ ở SciFact (+3.8%), SCIDOCS (+3.6%); đi ngang ở NFCorpus/FiQA/ArguAna. IDF vượt cả Official ở SciFact, SCIDOCS và FiQA.
- **Adaptive (chỉ std-dev) làm giảm chất lượng ở cả 5 tập**, nặng nhất SCIDOCS (−7.9%) và FiQA (−5.6%). Giả thuyết: chỉ có 200 candidate/query nên std-dev trên 200 giá trị quá nhiễu để ước lượng độ quan trọng của token.
- **CAD (IDF × std-dev) không vượt IDF ở bất kỳ tập nào.** Nó chỉ nhỉnh hơn Mini Base ở SciFact (+1.3%) và ArguAna (+0.6%, trong khoảng nhiễu); các tập còn lại đều thấp hơn Base.
- Ở FiQA, CAD (0.3214) bị kéo xuống gần bằng Adaptive (0.3209) và cách xa IDF (0.3414) → thành phần β đang **phá** tín hiệu IDF, chứ không bổ sung.
- → Trước khi kết luận về CAD cần: (1) **grid-search (α, β)** thay vì cố định α=β=1, và (2) thử **tăng `candidate_topn`** (200 → 500/1000) xem trần candidate hiện tại có đang chặn lợi ích.

### 7.6 SciFact (chi tiết, 300 test queries)

| Biến thể | nDCG@10 | MRR@10 | Recall@100 |
|---|---|---|---|
| Official ColBERTv2 | 0.6438 | 0.6120 | 0.8842 |
| Mini Base (2-bit IVF) | 0.6411 | 0.6103 | 0.8916 |
| IDF (α=1, β=0) | **0.6656** | **0.6367** | 0.8993 |
| Adaptive (α=0, β=1) | 0.6210 | 0.5903 | 0.8953 |
| CAD (α=1, β=1) | 0.6495 | 0.6204 | 0.8987 |

---

## 8. Dữ liệu

SciFact (và các tập BEIR khác như `nfcorpus`, `scidocs`, `fiqa`, `trec-covid`, `arguana`) được tải tự động bởi `encode.py` qua BEIR và lưu trong `data/`. Vì corpus, embeddings và index khá nặng, chúng bị loại khỏi git (xem `.gitignore`). Người dùng cần tự chạy pipeline để sinh ra.

Để chạy nhiều dataset trên Colab, dùng notebook [`Mini_ColBERTv2_MultiBEIR.ipynb`](Mini_ColBERTv2_MultiBEIR.ipynb); để có baseline official, dùng thêm [`Official_ColBERTv2_MultiBEIR.ipynb`](Official_ColBERTv2_MultiBEIR.ipynb).

---

## 9. License

Code trong repo này cung cấp cho mục đích học tập - nghiên cứu. Checkpoint ColBERT từ [colbert-ir/colbertv2.0](https://huggingface.co/colbert-ir/colbertv2.0) và tập SciFact từ BEIR.
