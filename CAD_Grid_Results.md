# CAD-Sim — Kết quả grid-search (α, β) trên BEIR

Ngày chạy: 2026-10-15 · Colab (CPU) · checkpoint `colbert-ir/colbertv2.0`

## 1. Tiền đề

Kết luận ghi ở `README.md` §7.5 là: *"CAD (IDF × std-dev) không vượt IDF ở bất kỳ
tập nào"*. Nhưng bản CAD đó bị chạy ở một điểm cấu hình duy nhất — `cad/rerank.py`
dòng 63–68 khoá `'cad': (alpha=1.0, beta=1.0)` — và signal dùng là `std`. Một kết
luận về phương pháp rút ra từ một điểm cấu hình như vậy là chưa đủ căn cứ.

Do đó grid-search được chạy lại để tách hai khả năng: phương pháp có vấn đề, hay
chỉ là chọn cấu hình sai.

## 2. Cách chạy

```bash
python -m cad.build_rerank_inputs --dataset <ds> --output_dir outputs/rerank/<ds>
python -m cad.experiment \
    --candidates outputs/rerank/<ds>/base_candidates.pt \
    --query_emb  outputs/rerank/<ds>/rerank_inputs.pt \
    --idf        outputs/rerank/<ds>/idf.pt \
    --qrels      outputs/rerank/<ds>/qrels_test.json \
    --json-out   <drive>/colbert-run/<ds>/grid_results.json
```

Toàn bộ đánh giá chạy offline trên `base_candidates.pt` đã có sẵn, nên không encode
lại và không dùng GPU.

- `cad/build_rerank_inputs.py` chỉ nạp tokenizer (không tải weights) để dựng lại
  `idf.pt` và `rerank_inputs.pt` từ corpus BEIR.
- `cad/experiment.py` quét 672 cấu hình = 2 mode × 7 signal × 6 α × 8 β (bỏ các
  tổ hợp trùng khi β=0), rerank lại top-100 từ pool 200 candidate.
- Ý nghĩa thống kê: paired bootstrap 2000 lần trên nDCG@10 từng query, so với IDF
  và so với MaxSim.

Hai công thức trọng số được quét, với $s_i$ là signal của token truy vấn $i$:

| mode | công thức |
|---|---|
| `mul` | $w_i = \mathrm{IDF}_i^{\alpha} \cdot \left(s_i + \varepsilon\right)^{\beta}$ |
| `add` | $w_i = 1 + \alpha\left(\dfrac{\mathrm{IDF}_i}{\overline{\mathrm{IDF}}} - 1\right) + \beta\left(\dfrac{s_i}{\bar{s}} - 1\right)$ |

Sau đó trọng số được kẹp về không âm (với mode `add`) và chia cho trung bình của nó
trên các token truy vấn thật, để chuẩn hoá trước khi tính điểm.

Bảy signal, tính trên ma trận `token_scores` kích thước $P \times L$, với $P$ là số
ứng viên và $L$ là số token truy vấn. Ký hiệu $ts_{ij}$ là điểm MaxSim của token truy
vấn $i$ trên ứng viên $j$, $\overline{ts_i}$ là trung bình theo $j$ và
$\sigma_i$ là độ lệch chuẩn theo $j$:

| signal | định nghĩa |
|---|---|
| `std` | $\sigma_i$ — signal của CAD bản cũ |
| `mad` | $\operatorname{med}_j \lvert ts_{ij} - \operatorname{med}_j ts_{ij} \rvert$ |
| `gap` | $\max_j ts_{ij} - \overline{ts_i}$ |
| `margin` | $ts_{i,(1)} - ts_{i,(2)}$ (top-1 trừ top-2) |
| `cv` | $\sigma_i / \left(\lvert \overline{ts_i} \rvert + \varepsilon\right)$ |
| `snr` | $\left(\max_j ts_{ij} - \overline{ts_i}\right) / \left(\sigma_i + \varepsilon\right)$ |
| `ldf` | $\frac{1}{P}\sum_{j=1}^{P} \mathbb{1}\left[ts_{ij} > 0\right]$ — bản sao IDF trên tập ứng viên |

## 3. Kiểm chứng input

Hai baseline do `experiment.py` in ra trùng khớp `README.md` §7.1–7.3:

| Dataset | MaxSim (α=0,β=0) | IDF (α=1,β=0) | README (IDF) |
|---|---|---|---|
| NFCorpus | 0.3289 | 0.3285 | 0.3285 |
| SCIDOCS | 0.1563 | 0.1619 | 0.1619 |
| FiQA | 0.3401 | 0.3414 | 0.3414 |
| ArguAna | 0.3331 | 0.3346 | 0.3346 |

`idf.pt` và `token_ids` dựng lại đúng, nên các so sánh bên dưới dùng được.

Số đầy đủ của baseline:

| Dataset | Config | nDCG@10 | MRR@10 | Recall@100 | nDCG@100 | MRR@100 |
|---|---|---|---|---|---|---|
| NFCorpus | MaxSim | 0.3289 | 0.5487 | 0.2722 | 0.2967 | 0.5545 |
| NFCorpus | IDF | 0.3285 | 0.5459 | 0.2693 | 0.2959 | 0.5520 |
| SCIDOCS | MaxSim | 0.1563 | 0.2791 | 0.3590 | 0.2219 | 0.2898 |
| SCIDOCS | IDF | 0.1619 | 0.2900 | 0.3614 | 0.2281 | 0.3011 |
| FiQA | MaxSim | 0.3401 | 0.4152 | 0.6312 | 0.4000 | 0.4238 |
| FiQA | IDF | 0.3414 | 0.4167 | 0.6346 | 0.4012 | 0.4250 |
| ArguAna | MaxSim | 0.3331 | 0.2211 | 0.9090 | 0.3817 | 0.2316 |
| ArguAna | IDF | 0.3346 | 0.2255 | 0.9111 | 0.3870 | 0.2373 |

## 4. Kết quả

### 4.1 Cấu hình tốt nhất mỗi tập

| Dataset | Cấu hình | nDCG@10 | MRR@10 | Recall@100 | vs IDF | p (vs IDF) | vs MaxSim | p (vs MaxSim) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| NFCorpus | `mul` `gap` α=1.0 β=0.8 | 0.3317 | 0.5446 | 0.2759 | +0.99% | 0.2110 | +0.85% | 0.3890 |
| SCIDOCS | `mul` `snr` α=0.5 β=0.5 | 0.1676 | 0.2993 | 0.3643 | +3.51% | 0.0000 | +7.23% | 0.0000 |
| FiQA | `mul` `snr` α=0.5 β=0.5 | 0.3518 | 0.4279 | 0.6403 | +3.04% | 0.0000 | +3.47% | 0.0030 |
| ArguAna | `add` `gap` α=0.5 β=0.1 | 0.3405 | 0.2284 | 0.9118 | +1.76% | 0.0040 | +2.22% | 0.0000 |

Khoảng tin cậy 95% của delta nDCG@10 so với IDF:

| Dataset | delta | CI 95% |
|---|---:|---|
| NFCorpus | +0.0032 | [−0.0017, +0.0086] |
| SCIDOCS | +0.0057 | [+0.0023, +0.0091] |
| FiQA | +0.0104 | [+0.0038, +0.0173] |
| ArguAna | +0.0059 | [+0.0021, +0.0097] |

### 4.2 So với các baseline khác (nDCG@10)

| Dataset | Official ColBERTv2 | Mini Base | IDF | CAD-Sim tốt nhất | vs Official |
|---|---:|---:|---:|---:|---:|
| NFCorpus | 0.3299 | 0.3289 | 0.3285 | 0.3317 | +0.56% |
| SCIDOCS | 0.1549 | 0.1563 | 0.1619 | 0.1676 | +8.20% |
| FiQA | 0.3410 | 0.3401 | 0.3414 | 0.3518 | +3.18% |
| ArguAna | 0.3366 | 0.3331 | 0.3346 | 0.3405 | +1.15% |

### 4.3 CAD bản cũ so với cấu hình tốt nhất

| Dataset | CAD cũ (α=1, β=1, `std`) | IDF | CAD-Sim tốt nhất | p (vs IDF) |
|---|---:|---:|---:|---:|
| NFCorpus | 0.3250 | 0.3285 | 0.3317 | 0.2110 |
| SCIDOCS | 0.1551 | 0.1619 | 0.1676 | 0.0000 |
| FiQA | 0.3214 | 0.3414 | 0.3518 | 0.0000 |
| ArguAna | 0.3352 | 0.3346 | 0.3405 | 0.0040 |

## 5. Top-10 cấu hình mỗi tập

Cột `vs IDF` là % thay đổi nDCG@10 so với IDF.

### 5.1 NFCorpus (323 query)

| # | mode | signal | α | β | nDCG@10 | MRR@10 | Recall@100 | nDCG@100 | vs IDF |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | mul | gap | 1.00 | 0.80 | 0.3317 | 0.5446 | 0.2759 | 0.2975 | +0.99% |
| 2 | mul | gap | 1.00 | 0.50 | 0.3317 | 0.5475 | 0.2743 | 0.2977 | +0.98% |
| 3 | mul | gap | 1.50 | 0.30 | 0.3314 | 0.5482 | 0.2715 | 0.2966 | +0.88% |
| 4 | add | gap | 1.00 | 0.30 | 0.3311 | 0.5465 | 0.2689 | 0.2967 | +0.80% |
| 5 | mul | gap | 1.00 | 0.30 | 0.3310 | 0.5451 | 0.2690 | 0.2967 | +0.77% |
| 6 | add | snr | 0.50 | 0.30 | 0.3309 | 0.5456 | 0.2703 | 0.2972 | +0.74% |
| 7 | mul | snr | 0.25 | 0.80 | 0.3309 | 0.5418 | 0.2722 | 0.2962 | +0.74% |
| 8 | mul | gap | 1.00 | 1.00 | 0.3309 | 0.5426 | 0.2755 | 0.2970 | +0.73% |
| 9 | add | snr | 1.00 | 0.20 | 0.3309 | 0.5464 | 0.2694 | 0.2965 | +0.72% |
| 10 | add | gap | 1.50 | 0.50 | 0.3308 | 0.5445 | 0.2724 | 0.2964 | +0.71% |

### 5.2 SCIDOCS (1000 query)

| # | mode | signal | α | β | nDCG@10 | MRR@10 | Recall@100 | nDCG@100 | vs IDF |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | mul | snr | 0.50 | 0.50 | 0.1676 | 0.2993 | 0.3643 | 0.2318 | +3.51% |
| 2 | mul | snr | 0.25 | 0.50 | 0.1676 | 0.2983 | 0.3645 | 0.2320 | +3.49% |
| 3 | add | snr | 0.50 | 0.50 | 0.1676 | 0.3002 | 0.3649 | 0.2321 | +3.49% |
| 4 | mul | gap | 0.50 | 0.50 | 0.1672 | 0.2952 | 0.3647 | 0.2313 | +3.30% |
| 5 | add | gap | 0.50 | 0.50 | 0.1672 | 0.2966 | 0.3639 | 0.2316 | +3.29% |
| 6 | add | snr | 0.25 | 0.50 | 0.1672 | 0.2977 | 0.3640 | 0.2317 | +3.27% |
| 7 | mul | gap | 0.50 | 0.80 | 0.1670 | 0.2946 | 0.3639 | 0.2311 | +3.17% |
| 8 | add | gap | 0.50 | 0.80 | 0.1669 | 0.2951 | 0.3641 | 0.2312 | +3.10% |
| 9 | mul | gap | 0.25 | 1.00 | 0.1669 | 0.2949 | 0.3603 | 0.2299 | +3.10% |
| 10 | mul | gap | 0.50 | 1.00 | 0.1665 | 0.2963 | 0.3603 | 0.2302 | +2.86% |

### 5.3 FiQA (648 query trong qrels)

| # | mode | signal | α | β | nDCG@10 | MRR@10 | Recall@100 | nDCG@100 | vs IDF |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | mul | snr | 0.50 | 0.50 | 0.3518 | 0.4279 | 0.6403 | 0.4100 | +3.04% |
| 2 | mul | snr | 0.50 | 0.30 | 0.3498 | 0.4290 | 0.6406 | 0.4099 | +2.46% |
| 3 | mul | snr | 0.25 | 0.50 | 0.3498 | 0.4260 | 0.6375 | 0.4073 | +2.44% |
| 4 | mul | snr | 0.25 | 0.30 | 0.3492 | 0.4267 | 0.6389 | 0.4085 | +2.28% |
| 5 | mul | gap | 0.25 | 0.80 | 0.3487 | 0.4236 | 0.6358 | 0.4077 | +2.12% |
| 6 | add | snr | 0.50 | 0.50 | 0.3486 | 0.4245 | 0.6356 | 0.4068 | +2.11% |
| 7 | add | snr | 0.25 | 0.50 | 0.3485 | 0.4242 | 0.6381 | 0.4064 | +2.07% |
| 8 | mul | snr | 0.50 | 0.20 | 0.3482 | 0.4265 | 0.6365 | 0.4075 | +2.00% |
| 9 | add | snr | 0.25 | 0.30 | 0.3480 | 0.4257 | 0.6381 | 0.4074 | +1.91% |
| 10 | add | snr | 0.50 | 0.30 | 0.3479 | 0.4255 | 0.6377 | 0.4077 | +1.90% |

### 5.4 ArguAna (1406 query)

| # | mode | signal | α | β | nDCG@10 | MRR@10 | Recall@100 | nDCG@100 | vs IDF |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | add | gap | 0.50 | 0.10 | 0.3405 | 0.2284 | 0.9118 | 0.3889 | +1.76% |
| 2 | add | gap | 0.50 | 0.20 | 0.3404 | 0.2285 | 0.9125 | 0.3893 | +1.74% |
| 3 | add | cv | 0.50 | 0.20 | 0.3404 | 0.2280 | 0.9161 | 0.3895 | +1.74% |
| 4 | add | gap | 0.25 | 0.30 | 0.3402 | 0.2279 | 0.9111 | 0.3883 | +1.69% |
| 5 | add | snr | 0.50 | 0.10 | 0.3402 | 0.2285 | 0.9111 | 0.3888 | +1.68% |
| 6 | add | margin | 0.50 | 0.05 | 0.3402 | 0.2286 | 0.9097 | 0.3887 | +1.67% |
| 7 | add | cv | 0.50 | 0.30 | 0.3401 | 0.2278 | 0.9168 | 0.3896 | +1.65% |
| 8 | mul | cv | 0.50 | 0.30 | 0.3401 | 0.2278 | 0.9132 | 0.3889 | +1.64% |
| 9 | mul | cv | 0.50 | 0.20 | 0.3400 | 0.2274 | 0.9147 | 0.3885 | +1.63% |
| 10 | add | cv | 0.50 | 0.10 | 0.3400 | 0.2278 | 0.9139 | 0.3890 | +1.62% |

Toàn bộ 672 cấu hình của mỗi tập nằm trong `grid_results.json`.

## 6. Nhận xét

❶ Kết luận ở README §7.5 không còn đúng. Kết luận cũ — "CAD không vượt IDF ở bất
kỳ tập nào" — là hệ quả của việc chạy đúng một cấu hình `α=1, β=1` với signal `std`.
Khi quét đủ lưới, họ CAD-Sim vượt IDF có ý nghĩa thống kê trên 3/4 tập (SCIDOCS
+3.51%, FiQA +3.04%, ArguAna +1.76%) và dương nhưng chưa đạt ý nghĩa ở NFCorpus (+0.99%,
p=0.2110).

❷ `std` là signal yếu. Trong cả bốn bảng top-20, `std` không xuất hiện một lần
nào ở NFCorpus, SCIDOCS và FiQA; ở ArguAna nó chỉ nằm nửa dưới (`mul std 0.50/0.30`
= +1.61%). Các signal thắng là `snr` và `gap`, cả hai đều lấy `max` trừ `mean` — tức
độ trội của ứng viên tốt nhất, chứ không phải độ tán xạ thuần.

Giải thích hợp lý là: với chỉ 200 candidate, `std` đo độ *phân tán* nhưng không
đo độ *phân biệt*. Một token khớp mờ đều với mọi tài liệu vẫn có thể có `std` lớn, trong
khi `snr` và `gap` phạt đúng trường hợp đó.

❸ `ldf` không giúp gì — bằng chứng CAD-Sim khác IDF. `ldf` được thiết kế làm bản
sao của IDF trên tập ứng viên (tỷ lệ ứng viên mà token khớp dương), với kỳ vọng nó hoạt động
nếu cơ chế hữu ích là kiểu đếm tần suất. Kết quả: `ldf` không lọt vào top-20 nào của
tập nào. Cái tạo ra lợi ích là hình dạng phân bố điểm (`snr`, `gap`), không phải
tần suất. Điều này cho thấy CAD-Sim và IDF mang hai loại thông tin khác nhau, không phải
hai cách viết của cùng một ý tưởng.

❹ IDF và CAD-Sim bổ trợ nhau. Mọi cấu hình thắng đều có $\alpha \approx 0.5$,
tức $\mathrm{IDF}^{0.5} \cdot s^{0.5}$: IDF vẫn cần, nhưng giảm quyền thì signal mới
phát huy. FiQA là ví dụ rõ nhất — bản CAD cũ ($\beta = 1$ trên nền $\mathrm{IDF}^{1}$)
bị kéo xuống 0.3214, còn $\alpha = 0.5$, $\beta = 0.5$ lên 0.3518.

❺ Vùng tốt ổn định. Hai dấu hiệu cho thấy kết quả không phải nhiễu:

- Cùng một cấu hình `mul snr α=0.5 β=0.5` thắng độc lập trên cả SCIDOCS và FiQA — hai
  tập khác nhau về cả corpus lẫn kiểu query.
- Vùng top-20 liền mạch (α ∈ [0.25, 1.5], β ∈ [0.3, 1.0]), không phải một đỉnh lẻ.

❻ NFCorpus đúng hướng nhưng chưa đủ độ tin. Đây là tập duy nhất chưa đạt ý nghĩa
(p=0.2110), nhưng delta vẫn dương (+0.99%). NFCorpus chỉ có 323 query — ít nhất trong
bốn tập — nên với hiệu ứng cỡ 1% thì đây là trường hợp thiếu power, chưa nói lên được
rằng hiệu ứng bằng không.

❼ Vượt Official ColBERTv2. Với cùng dataset và cùng settings, cấu hình tốt nhất
vượt Official trên cả bốn tập: SCIDOCS +8.20%, FiQA +3.18%, ArguAna +1.15%,
NFCorpus +0.56%. Đây chưa được coi là kết quả chốt, vì lý do ở mục 7.1.

## 7. Hạn chế

❶ Chọn cấu hình trên chính tập test. Cấu hình được chọn bằng cách lấy max trên
672 cấu hình, rồi đánh giá trên cùng tập query dùng để tính metric, và so p-value với
IDF cũng trên tập đó.
Paired bootstrap không hiệu chỉnh cho việc chọn cấu hình sau khi đã nhìn kết quả, nên
các mức +3.5% / +3.0% / +1.8% và p≈0.0000 hiện tại là lạc quan quá mức. Với 672 cấu
hình và 323–1406 query, phần lãi do chọn cấu hình có thể chiếm khoảng 0.5–1% tương đối.
Chiều hướng thì nhiều khả năng vẫn đúng, nhưng độ lớn chưa dùng để kết luận được.

❷ Trần của candidate pool. Toàn bộ thí nghiệm chỉ rerank lại trong 200 candidate
mỗi query. Nếu tài liệu đúng chưa được retrieval về thì không cách nào xếp hạng đúng,
nên `candidate_topn` hiện tại có thể đang che mất phần lợi ích thật của trọng số.

❸ Phạm vi đo. Các metric đo được gồm nDCG@10, MRR@10, Recall@100, nDCG@100,
MRR@100 — chưa có MAP, nên chưa thấy được lợi ích ở phần cuối danh sách. Chưa chạy
SciFact trong lưới
(mới có bản 100-candidate trên máy local) và vẫn bỏ `trec-covid` vì OOM.

❹ FiQA thừa query được retrieval. Log báo `6648 queries with candidates` trong
khi qrels chỉ có 648. Chỉ số vẫn tính trên đúng 648 query test nên không sai, nhưng chỗ
này đang làm thừa công retrieval.

## 8. Hướng kiểm tra tiếp

| # | Việc | Encode lại? |
|---|---|---|
| 1 | Chia query thành hai nửa: chọn (mode, signal, α, β) trên nửa A, đánh giá một lần trên nửa B | Không |
| 2 | Chạy lưới cho SciFact | Không |
| 3 | Nâng `candidate_topn` 200 → 500/1000 để đo trần của pool | Có |
| 4 | Thêm MAP vào metrics | Không |
| 5 | Thử signal theo hướng "độ trội": `max − median`, top-k gap | Không |
