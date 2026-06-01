# HƯỚNG DẪN TÁI HIỆN VÀ DEMO TOÀN BỘ QUY TRÌNH TOFU BENCHMARK (MÔ HÌNH PHI-1.5)

Tài liệu này cung cấp hướng dẫn chi tiết từng bước để chạy thực nghiệm unlearning và đánh giá (evaluation) mô hình **Phi-1.5** theo đúng phương pháp của bài báo khoa học TOFU trên môi trường **Kaggle Notebook (2x GPU T4)**.

---

## BƯỚC 1: Chuẩn bị Môi trường, Tải mã nguồn và Cài đặt Thư viện
Trong cell đầu tiên của Kaggle Notebook, chuyển cấu hình **Accelerator** sang **GPU T4 x2**, sau đó chạy các lệnh sau để tải mã nguồn từ GitHub cá nhân của bạn, di chuyển vào thư mục làm việc và cài đặt các thư viện cần thiết:

```python
# 1. Tải mã nguồn của bạn về Kaggle (Hãy thay thế bằng link GitHub của bạn)
!git clone https://github.com/TÊN_GITHUB_CỦA_BẠN/tofu.git

# 2. Di chuyển thư mục làm việc vĩnh viễn vào trong repo vừa tải về
%cd tofu

# 3. Cài đặt các thư viện cần thiết từ Hugging Face và các package bổ trợ
!pip install -q git+https://github.com/huggingface/transformers
!pip install -q datasets accelerate deepspeed evaluate peft rouge_score hydra-core omegaconf bitsandbytes scipy ninja natsort matplotlib
```

* **Lưu ý về mô hình đã finetune gốc:** Bài báo sử dụng mô hình `locuslab/tofu_ft_phi-1.5` đã được finetune sẵn trên toàn bộ 4.000 câu hỏi (200 tác giả hư cấu) có sẵn trên Hugging Face. Chúng ta sẽ sử dụng trực tiếp mô hình này để tiết kiệm thời gian huấn luyện ban đầu.

---

## BƯỚC 2: Đánh giá Mô hình Tham chiếu (Retain-Only Model)
Để thực hiện phép kiểm định thống kê **KS-Test (Forget Quality)**, bài báo yêu cầu so sánh phân phối xác suất của mô hình Unlearn với một mô hình "lý tưởng" (Oracle) — mô hình chỉ được học tập Retain (90% dữ liệu còn lại) mà chưa từng nhìn thấy 10% dữ liệu cần xóa.

Mô hình lý tưởng này có sẵn trên Hugging Face dưới tên `locuslab/tofu_ft_phi-1.5_retain90`. Chạy lệnh dưới đây để trích xuất kết quả đánh giá tham chiếu:

```bash
!python evaluate_util.py \
    model_family=phi \
    use_pretrained=true \
    model_path=locuslab/tofu_ft_phi-1.5_retain90 \
    save_dir=eval_results/phi_retain90 \
    batch_size=16
```
*Kết quả đánh giá sẽ được lưu tại: `eval_results/phi_retain90/checkpoint-0/eval_log_aggregated.json`.*

---

## BƯỚC 3: Thực hiện Unlearning với 4 Thuật toán
Chúng ta sẽ chạy song song 4 thuật toán unlearning trên **2x GPU T4** để xóa **10% dữ liệu (`forget10`)** trong 5 epochs sử dụng `accelerate launch` để phân mảnh mô hình qua DeepSpeed Stage 3:

### 1. Thuật toán Gradient Ascent (GA)
```bash
!accelerate launch --multi_gpu --num_processes=2 forget.py \
    model_family=phi \
    forget_loss=grad_ascent \
    split=forget10 \
    batch_size=8 \
    gradient_accumulation_steps=4 \
    lr=1e-5 \
    num_epochs=5 \
    save_model=true \
    overwrite_dir=true \
    save_dir=models/phi_unlearn_GA
```

### 2. Thuật toán Gradient Difference (GD)
```bash
!accelerate launch --multi_gpu --num_processes=2 forget.py \
    model_family=phi \
    forget_loss=grad_diff \
    split=forget10 \
    batch_size=8 \
    gradient_accumulation_steps=4 \
    lr=1e-5 \
    num_epochs=5 \
    save_model=true \
    overwrite_dir=true \
    save_dir=models/phi_unlearn_GD
```

### 3. Thuật toán KL Minimization (KL)
```bash
!accelerate launch --multi_gpu --num_processes=2 forget.py \
    model_family=phi \
    forget_loss=KL \
    split=forget10 \
    batch_size=8 \
    gradient_accumulation_steps=4 \
    lr=1e-5 \
    num_epochs=5 \
    save_model=true \
    overwrite_dir=true \
    save_dir=models/phi_unlearn_KL
```

### 4. Thuật toán Preference Optimization (DPO)
```bash
!accelerate launch --multi_gpu --num_processes=2 forget.py \
    model_family=phi \
    forget_loss=dpo \
    split=forget10 \
    batch_size=8 \
    gradient_accumulation_steps=4 \
    lr=1e-5 \
    num_epochs=5 \
    save_model=true \
    overwrite_dir=true \
    save_dir=models/phi_unlearn_DPO
```

---

## BƯỚC 4: Đánh giá Mô hình sau khi Unlearn
Sau khi cả 4 tiến trình unlearn hoàn tất, chạy đánh giá độc lập cho từng mô hình đã xóa dữ liệu để trích xuất file JSON kết quả:

```bash
# Đánh giá Gradient Ascent
!python evaluate_util.py model_family=phi model_path=models/phi_unlearn_GA save_dir=eval_results/phi_unlearn_GA batch_size=16

# Đánh giá Gradient Difference
!python evaluate_util.py model_family=phi model_path=models/phi_unlearn_GD save_dir=eval_results/phi_unlearn_GD batch_size=16

# Đánh giá KL Minimization
!python evaluate_util.py model_family=phi model_path=models/phi_unlearn_KL save_dir=eval_results/phi_unlearn_KL batch_size=16

# Đánh giá DPO
!python evaluate_util.py model_family=phi model_path=models/phi_unlearn_DPO save_dir=eval_results/phi_unlearn_DPO batch_size=16
```

---

## BƯỚC 5: Tính toán Model Utility và Forget Quality (KS-Test)
Sử dụng tệp `aggregate_eval_stat.py` để so sánh kết quả đánh giá của từng phương pháp với mô hình lý tưởng Retain-Only để tính toán **Forget Quality** (p-value của kiểm định Kolmogorov-Smirnov) và **Model Utility**:

```bash
# Tính toán thống kê cho Gradient Ascent
!python aggregate_eval_stat.py \
    retain_result=eval_results/phi_retain90/checkpoint-0/eval_log_aggregated.json \
    ckpt_result=eval_results/phi_unlearn_GA/checkpoint-0/eval_log_aggregated.json \
    method_name=grad_ascent \
    submitted_by=Group5 \
    save_file=eval_results/stat_GA.csv

# Tính toán thống kê cho Gradient Difference
!python aggregate_eval_stat.py \
    retain_result=eval_results/phi_retain90/checkpoint-0/eval_log_aggregated.json \
    ckpt_result=eval_results/phi_unlearn_GD/checkpoint-0/eval_log_aggregated.json \
    method_name=grad_diff \
    submitted_by=Group5 \
    save_file=eval_results/stat_GD.csv

# Tính toán thống kê cho KL Minimization
!python aggregate_eval_stat.py \
    retain_result=eval_results/phi_retain90/checkpoint-0/eval_log_aggregated.json \
    ckpt_result=eval_results/phi_unlearn_KL/checkpoint-0/eval_log_aggregated.json \
    method_name=KL \
    submitted_by=Group5 \
    save_file=eval_results/stat_KL.csv

# Tính toán thống kê cho DPO
!python aggregate_eval_stat.py \
    retain_result=eval_results/phi_retain90/checkpoint-0/eval_log_aggregated.json \
    ckpt_result=eval_results/phi_unlearn_DPO/checkpoint-0/eval_log_aggregated.json \
    method_name=DPO \
    submitted_by=Group5 \
    save_file=eval_results/stat_DPO.csv
```

---

## BƯỚC 6: Trực quan hóa Biểu đồ (In đồ thị y hệt bài báo)
Sử dụng tệp script vẽ đồ thị được tích hợp sẵn [plot_results.py](file:///c:/Users/Admin/Documents/A-DO_AN_ML/tofu/plot_results.py) trong workspace để tự động kết xuất ra biểu đồ.

Bạn sao chép đoạn mã Python dưới đây vào một cell trong Kaggle Notebook để vẽ:

```python
import csv
import plot_results

# 1. Vẽ biểu đồ so sánh chỉ số ROUGE-L, Probability, Truth Ratio trên 4 tập dữ liệu
methods = ['GA', 'GD', 'KL', 'DPO']
method_names = ['Gradient Ascent', 'Gradient Difference', 'KL Minimization', 'Preference Opt (DPO)']
method_summaries = {}

for m, name in zip(methods, method_names):
    file_path = f"eval_results/phi_unlearn_{m}/checkpoint-0/eval_log_aggregated.json"
    data = plot_results.load_aggregated_json(file_path)
    if data:
        method_summaries[name] = plot_results.get_metrics_summary(data)

# Đọc thêm kết quả của Retain-Only Model (Oracle) để so sánh đối chiếu
retain_data = plot_results.load_aggregated_json("eval_results/phi_retain90/checkpoint-0/eval_log_aggregated.json")
if retain_data:
    method_summaries['Retain-Only (Oracle)'] = plot_results.get_metrics_summary(retain_data)

plot_results.plot_unlearn_curves(method_summaries, save_path="unlearn_curves.png")

# 2. Vẽ biểu đồ Trade-off giữa Model Utility (Trục X) và Forget Quality (Trục Y)
methods_tradeoff = []
for m, name in zip(methods, method_names):
    csv_path = f"eval_results/stat_{m}.csv"
    with open(csv_path, mode='r') as infile:
        reader = csv.DictReader(infile)
        for row in reader:
            methods_tradeoff.append({
                'name': name,
                'model_utility': float(row['Model Utility']),
                'forget_quality': float(row['Forget Quality'])
            })

# Điểm chuẩn của mô hình lý tưởng Oracle
methods_tradeoff.append({
    'name': 'Oracle Baseline',
    'model_utility': 1.0,
    'forget_quality': 1.0
})

plot_results.plot_tradeoff_scatter(methods_tradeoff, save_path="tradeoff_scatter.png")
```

---

## HƯỚNG DẪN TRÌNH DIỄN VÀ BÁO CÁO KẾT QUẢ

Khi báo cáo đồ án, nhóm bạn nên sử dụng các kết quả trực quan hóa từ hai biểu đồ trên để làm nổi bật chiều sâu khoa học:

1. **Phân tích Biểu đồ so sánh (`unlearn_curves.png`)**:
   * Chỉ ra rằng đối với tập **Forget Set**, điểm số ROUGE-L và Probability của mô hình Unlearn bị giảm mạnh (chứng minh việc unlearn hoạt động tốt).
   * Tuy nhiên, trên tập **Retain Set** và **World Facts**, chỉ số của các phương pháp nâng cao (KL, DPO) phải giữ ở mức cao, tiệm cận với đường màu của Oracle. Trong khi đó, **Gradient Ascent (GA)** sẽ bị sụt giảm mạnh ở cả các vùng tri thức này (chứng minh GA xóa dữ liệu một cách cực đoan và phá hủy tri thức lành mạnh).

2. **Phân tích Biểu đồ Đánh đổi (`tradeoff_scatter.png`)**:
   * Mục tiêu lý tưởng là tiệm cận điểm **Oracle Baseline (1.0, 1.0)** ở góc trên bên phải.
   * **Gradient Ascent (GA)** thường nằm ở góc trên bên trái (Forget Quality cao nhưng Model Utility cực thấp).
   * **KL Minimization** và **DPO** sẽ nằm gần góc trên bên phải nhất, chứng minh đây là các phương pháp tối ưu giúp cân bằng xuất sắc giữa chất lượng xóa tri thức mục tiêu và bảo toàn tri thức chung.
