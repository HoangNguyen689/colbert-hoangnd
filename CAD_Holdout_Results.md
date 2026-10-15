# CAD-Sim — Kiểm định hold-out

Ngày đo: 2026-10-15 · Colab (CPU) · checkpoint `colbert-ir/colbertv2.0`

## 1. Lý do

Mục 7.1 của [`CAD_Grid_Results.md`](CAD_Grid_Results.md) đã ghi nhận một hạn chế:
cấu hình tốt nhất được chọn bằng cách lấy max trên 672 cấu hình, rồi đánh giá trên
chính tập query dùng để tính metric, và p-value cũng tính trên tập đó. Paired
bootstrap không hiệu chỉnh cho việc chọn cấu hình sau khi đã nhìn kết quả, nên các
mức lợi ích ở đó là lạc quan quá mức.

Kiểm định này đo lại theo giao thức giữ lại một nửa query, để tách phần lợi ích thật
khỏi phần lợi ích sinh ra do chọn cấu hình.

## 2. Giao thức

Chương trình dùng cho kiểm định này là `cad/experiment_holdout.py`. Với mỗi seed:

1. Chia query. Lấy các qid có trong `qrels`, sắp xếp cố định, rồi sinh một hoán vị bằng
   `numpy.random.default_rng(seed)`. Nửa A gồm $n_A$ qid đầu của hoán vị, với
   $n_A = \lceil \lvert Q \rvert \cdot \text{frac} \rceil$; nửa B là phần còn lại. Hai
   nửa rời nhau hoàn toàn: $A \cap B = \emptyset$. Cùng seed cho cùng cách chia trên mọi máy.
2. Quét lưới trên A. Toàn bộ cấu hình được chấm trên nửa A, chọn ra $\text{best}_A$
   là cấu hình có nDCG@10 cao nhất trên A. Nửa B chưa được dùng tới.
3. Chấm $\text{best}_A$ trên B. Cấu hình đã chọn được chạy trên nửa B, cùng với
   baseline IDF trên nửa B.
4. So sánh trên B. Paired bootstrap trên nDCG@10 từng query giữa $\text{best}_A$ và
   IDF, cả hai đều trên B.
5. Mốc trần. Quét lưới trên chính nửa B và lấy max, ký hiệu $\text{oracle}_B$. Đây
   là điểm cao nhất mà một cấu hình bất kỳ trong lưới đạt được trên nửa B, tức được phép
   nhìn B để chọn. Không dùng được trong thực tế, nhưng cho biết nửa B có bao nhiêu "đất".
6. Lặp qua các seed và lấy trung bình.

Hai đại lượng báo cáo:

$$\text{gain} = \frac{\mathrm{nDCG}_{10}^{B}(\text{best}_A) - \mathrm{nDCG}_{10}^{B}(\mathrm{IDF})}{\mathrm{nDCG}_{10}^{B}(\mathrm{IDF})} \times 100$$

$$\text{oracle gain} = \frac{\mathrm{nDCG}_{10}^{B}(\text{oracle}_B) - \mathrm{nDCG}_{10}^{B}(\mathrm{IDF})}{\mathrm{nDCG}_{10}^{B}(\mathrm{IDF})} \times 100$$

Khoảng cách giữa hai giá trị trung bình là phần lợi ích đến từ việc chọn cấu hình.

Lệnh chạy, giống nhau cho cả bốn tập:

```bash
python -m cad.experiment_holdout \
    --candidates outputs/rerank/<ds>/base_candidates.pt \
    --query_emb  outputs/rerank/<ds>/rerank_inputs.pt \
    --idf        outputs/rerank/<ds>/idf.pt \
    --qrels      outputs/rerank/<ds>/qrels_test.json \
    --signals std,gap,snr,cv --alphas 0,0.25,0.5,1,1.5 \
    --betas 0,0.1,0.3,0.5,0.8,1 --seeds 0,1,2 \
    --json-out   <drive>/colbert-run/<ds>/holdout_results.json
```

Lưới rút còn 209 cấu hình trên một tập (2 mode × 4 signal × 5 α × 6 β, trừ các tổ hợp
trùng khi $\beta = 0$ và trừ baseline). Lý do rút: mỗi seed quét lưới hai lần (A để chọn,
B để lấy mốc trần), nên một seed tốn xấp xỉ một lượt lưới đầy đủ. Ba seed tốn khoảng ba
lượt, nên lưới rút để tổng thời gian nằm trong khoảng chấp nhận được.

## 3. Kiểm chứng input

Hai baseline trên toàn tập do `experiment_holdout.py` in ra trùng khớp
`README.md` §7.1–7.3 và `CAD_Grid_Results.md` §3:

| Dataset | Config | nDCG@10 | MRR@10 | Recall@100 | nDCG@100 | MRR@100 |
|---|---|---|---|---|---|---|
| NFCorpus | MaxSim | 0.3289 | 0.5487 | 0.2722 | 0.2967 | 0.5545 |
| NFCorpus | IDF | 0.3285 | 0.5459 | 0.2693 | 0.2959 | 0.5520 |
| SCIDOCS | MaxSim | 0.1563 | 0.2791 | 0.3590 | 0.2219 | 0.2898 |
| SCIDOCS | IDF | 0.1619 | 0.2900 | 0.3614 | 0.2281 | 0.3011 |
| ArguAna | MaxSim | 0.3331 | 0.2211 | 0.9090 | 0.3817 | 0.2316 |
| ArguAna | IDF | 0.3346 | 0.2255 | 0.9111 | 0.3870 | 0.2373 |
| FiQA | MaxSim | 0.3401 | 0.4152 | 0.6312 | 0.4000 | 0.4238 |
| FiQA | IDF | 0.3414 | 0.4167 | 0.6346 | 0.4012 | 0.4250 |

## 4. Kết quả

### 4.1 Tóm tắt theo tập

| Dataset | Full-set (672 cfg) | Held-out | Oracle-B | Giữ được | Seed dương | Split đạt p<0.05 |
|---|---:|---:|---:|---:|---:|---:|
| NFCorpus | +0.99% (p=0.2110) | −0.89% | +1.09% | không giữ được | 0/3 | 0/3 |
| SCIDOCS | +3.51% (p=0.0000) | +1.91% | +4.17% | 54% | 3/3 | 2/3 |
| ArguAna | +1.76% (p=0.0040) | +1.15% | +1.93% | 65% | 3/3 | 0/3 |
| FiQA | +3.04% (p=0.0000) | +2.10% | +3.27% | 69% | 3/3 | 2/3 |

Cột Full-set là kết quả chọn cấu hình trên toàn tập, lấy từ
[`CAD_Grid_Results.md`](CAD_Grid_Results.md) §4.1. Cột Held-out là trung bình của ba
seed. Cột Giữ được là tỉ lệ giữa hai giá trị.

### 4.2 Chi tiết từng seed

NFCorpus (323 query, 162/161)

| Seed | Chọn trên A | nDCG@10 B | IDF B | gain | p | Oracle B | oracle gain |
|---:|---|---:|---:|---:|---:|---:|---:|
| 0 | `add` `snr` 0.5/0.8 | 0.3253 | 0.3292 | −1.17% | 0.3030 | 0.3341 | +1.50% |
| 1 | `mul` `gap` 0.5/1.0 | 0.3329 | 0.3331 | −0.05% | 0.9470 | 0.3363 | +0.97% |
| 2 | `mul` `snr` 0.25/0.8 | 0.2745 | 0.2786 | −1.45% | 0.2520 | 0.2808 | +0.80% |

Trung bình: gain −0.89%, oracle gain +1.09%, 0/3 split đạt p<0.05.

SCIDOCS (1000 query, 500/500)

| Seed | Chọn trên A | nDCG@10 B | IDF B | gain | p | Oracle B | oracle gain |
|---:|---|---:|---:|---:|---:|---:|---:|
| 0 | `mul` `gap` 0.5/0.5 | 0.1673 | 0.1626 | +2.89% | 0.0120 | 0.1688 | +3.83% |
| 1 | `add` `snr` 0.25/0.5 | 0.1608 | 0.1603 | +0.30% | 0.7910 | 0.1645 | +2.62% |
| 2 | `mul` `gap` 0.5/0.5 | 0.1656 | 0.1615 | +2.53% | 0.0390 | 0.1713 | +6.06% |

Trung bình: gain +1.91%, oracle gain +4.17%, 2/3 split đạt p<0.05.

ArguAna (1406 query, 703/703)

| Seed | Chọn trên A | nDCG@10 B | IDF B | gain | p | Oracle B | oracle gain |
|---:|---|---:|---:|---:|---:|---:|---:|
| 0 | `add` `gap` 0.25/0.5 | 0.3506 | 0.3484 | +0.61% | 0.4850 | 0.3530 | +1.30% |
| 1 | `add` `gap` 0.25/0.3 | 0.3438 | 0.3380 | +1.72% | 0.0890 | 0.3460 | +2.38% |
| 2 | `add` `gap` 0.5/0.5 | 0.3306 | 0.3269 | +1.12% | 0.1390 | 0.3338 | +2.10% |

Trung bình: gain +1.15%, oracle gain +1.93%, 0/3 split đạt p<0.05.

FiQA (648 query, 324/324)

| Seed | Chọn trên A | nDCG@10 B | IDF B | gain | p | Oracle B | oracle gain |
|---:|---|---:|---:|---:|---:|---:|---:|
| 0 | `mul` `snr` 0.5/0.3 | 0.3752 | 0.3674 | +2.11% | 0.0180 | 0.3795 | +3.29% |
| 1 | `mul` `snr` 0.5/0.5 | 0.3499 | 0.3401 | +2.88% | 0.0420 | 0.3502 | +2.98% |
| 2 | `add` `snr` 0.25/0.5 | 0.3372 | 0.3329 | +1.30% | 0.4300 | 0.3447 | +3.53% |

Trung bình: gain +2.10%, oracle gain +3.27%, 2/3 split đạt p<0.05.

## 5. Nhận xét

### 5.1 Lợi ích thật bằng một nửa đến hai phần ba con số công bố

So với kết quả chọn cấu hình trên toàn tập, mức giữ lại được là 54% ở SCIDOCS, 65% ở
ArguAna và 69% ở FiQA. Phần chênh còn lại là phần lợi ích sinh ra do đã nhìn vào tập
đang chấm điểm, không phải do trọng số.

### 5.2 NFCorpus: mức +0.99% trước đó không đứng vững

Cả ba seed đều âm (−1.17%, −0.05%, −1.45%) và không seed nào đạt ý nghĩa. Mức dương
+0.99% ghi ở `CAD_Grid_Results.md` §4.1 (p=0.2110) là hệ quả của việc chọn cấu hình
trên chính tập test, không phải một hiệu ứng nhỏ bị thiếu power.

Một chi tiết đáng chú ý: baseline IDF trên nửa B của ba seed là 0.3292, 0.3331 và
0.2786, trong khi trên toàn tập là 0.3285. Seed 2 lệch 15%, tức việc chia nửa trên
NFCorpus rất không ổn định. Ba seed là ít cho tập này, nhưng dấu âm ở cả ba vẫn nhất quán.

### 5.3 SCIDOCS và FiQA: hiệu ứng tồn tại, mức độ nhỏ hơn

- SCIDOCS: 3/3 seed dương, 2/3 đạt ý nghĩa (p=0.0120 và 0.0390), giữ 54%.
- FiQA: 3/3 seed dương, 2/3 đạt ý nghĩa (p=0.0180 và 0.0420), giữ 69%.

Đây là hai tập duy nhất mà hiệu ứng vừa giữ được độ lớn, vừa đạt ý nghĩa trên query
chưa từng dùng để chọn cấu hình.

### 5.4 ArguAna: đúng hướng nhưng chưa đủ lực ở nửa tập

3/3 seed dương (+0.61%, +1.72%, +1.12%) nhưng không seed nào đạt ý nghĩa. Trên toàn tập
1406 query, kết quả cũ đạt p=0.0040. Với 703 query mỗi nửa, độ lệch chuẩn của bootstrap
tăng lên nên hiệu ứng cỡ 1% không còn phân biệt được với 0. Đây là mất power do chia
đôi, không phải mất hiệu ứng.

### 5.5 Chọn cấu hình ổn định, và luôn rơi vào `gap` hoặc `snr`

12/12 lần chọn trên các nửa A đều rơi vào `gap` (6 lần) hoặc `snr` (6 lần):

| Dataset | Seed 0 | Seed 1 | Seed 2 |
|---|---|---|---|
| NFCorpus | `add snr` | `mul gap` | `mul snr` |
| SCIDOCS | `mul gap` | `add snr` | `mul gap` |
| ArguAna | `add gap` | `add gap` | `add gap` |
| FiQA | `mul snr` | `mul snr` | `add snr` |

Không lần nào chọn `std` hay `cv`. Vùng tham số thắng là α ∈ {0.25, 0.5} (0.5 xuất hiện
7/12 lần) và β ∈ {0.3, 0.5, 0.8, 1.0} (0.5 xuất hiện 7/12 lần). Đây trùng với vùng thắng
của lưới toàn tập, tức việc chọn trên nửa tập không nhảy loạn mà chỉ lấy hụt một phần.

### 5.6 Dấu của kết quả nhất quán trên cả ba tập có hiệu ứng

Chín trên chín split dương ở SCIDOCS, ArguAna và FiQA; ba trên ba split âm ở NFCorpus.
Nếu dấu là ngẫu nhiên thì xác suất ra 9/9 dương là khoảng 0.2%. Cần lưu ý ba tập này
không độc lập hoàn toàn với nhau, nên đây là dấu hiệu nhất quán chứ không phải một kiểm
định thống kê độc lập.

### 5.7 Mốc trần không phải mục tiêu

$\text{oracle}_B$ là max trên 209 cấu hình tính trên chính nửa B, nên bản thân nó cũng
bị thổi phồng do chọn cấu hình. Vì vậy không nên diễn giải theo hướng "giữ được 46% của
mốc trần". Cách dùng đúng: mốc trần cho biết nửa B có bao nhiêu đất, còn held-out cho
biết lấy được bao nhiêu trong số đó. Ở FiQA và SCIDOCS lấy được phần lớn; ở NFCorpus có
đất (+1.09%) mà không lấy nổi (−0.89%).

## 6. Hạn chế

### 6.1 Trung bình qua nhiều seed không phải một tập test sạch

Lấy trung bình ba seed khiến mọi query đều được dùng tới ở lần này hay lần khác. Đây là
repeated random subsampling, không phải một tập test để nguyên chưa từng chạm. Từng ước
lượng đơn lẻ không bị lệch, nhưng con số trung bình không còn là một phép đo "test sạch"
theo nghĩa chặt nhất. Muốn chặt hơn thì cố định một seed, chọn trên A, báo cáo trên B và
không nhìn B lần nữa; đổi lại chỉ có một con số và không có độ bất định.

### 6.2 Ba seed là ít

Các seed dao động khá ở một số tập: nhóm gain của SCIDOCS là +2.89%, +0.30%, +2.53% và
của NFCorpus là −1.17%, −0.05%, −1.45%. Trung bình ba giá trị còn nhiễu, đặc biệt trên
tập có ít query.

### 6.3 Chia nửa không ổn định như nhau ở mọi tập

Baseline IDF trên nửa B, so với IDF trên toàn tập:

| Dataset | IDF full | IDF B (3 seed) | Biên độ tương đối |
|---|---:|---|---:|
| NFCorpus | 0.3285 | 0.3292 / 0.3331 / 0.2786 | 16.6% |
| SCIDOCS | 0.1619 | 0.1626 / 0.1603 / 0.1615 | 1.4% |
| ArguAna | 0.3346 | 0.3484 / 0.3380 / 0.3269 | 6.4% |
| FiQA | 0.3414 | 0.3674 / 0.3401 / 0.3329 | 10.1% |

NFCorpus và FiQA có biên độ lớn, nên mức lợi ích đo trên nửa tập ở hai tập này nhiễu hơn
so với SCIDOCS.

### 6.4 Lưới rút gọn

Kiểm định dùng 209 cấu hình thay vì 672. Nếu mốc trần bị thấp ở một tập là do lưới thiếu
cấu hình tốt, chứ chưa chắc do trần candidate.

### 6.5 Trần candidate không đổi

Toàn bộ kiểm định vẫn chỉ rerank lại trong 200 candidate mỗi query, lấy từ pool có sẵn.
Lợi ích của trọng số vẫn bị chặn bởi pool này.

### 6.6 Chỉ đo nDCG@10

Paired bootstrap và việc chọn cấu hình đều dựa trên nDCG@10. MRR@10, Recall@100,
nDCG@100 và MRR@100 không được dùng trong kiểm định này.

## 7. Hướng kiểm tra tiếp

| # | Việc | Encode lại? |
|---|---|---|
| 1 | Chạy `--seeds 0,1,2,3,4` (5 seed) cho cả bốn tập để thu hẹp độ bất định | Không |
| 2 | Cố định một seed, chọn trên A và báo cáo một lần trên B, không nhìn B nữa | Không |
| 3 | Chạy kiểm định cho SciFact | Không |
| 4 | Nâng `candidate_topn` 200 → 500/1000 rồi kiểm định lại | Có |
| 5 | Đưa nDCG@100 vào tiêu chí chọn cấu hình, vì đây là chỉ số ổn định hơn nDCG@10 | Không |
| 6 | Cập nhật `CAD_Grid_Results.md` §4.1, §6 và §7.1 theo kết quả trong file này | Không |
