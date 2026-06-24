# Kế hoạch & Quy trình Thí nghiệm GUDS-EDL

Tài liệu này mô tả toàn bộ kế hoạch, quy trình cài đặt và thực thi các thí nghiệm nhằm đánh giá kiến trúc **GUDS-EDL (Generalized Uncertainty-Guided Dynamic Sparsification for Evidential Long-Tailed Learning)** trên các bộ dữ liệu có mức độ phân bố mất cân bằng nghiêm trọng (Long-Tailed / Extreme Imbalance).

---

## 1. Mục tiêu Thí nghiệm
- Chứng minh khả năng học hiệu quả trên dữ liệu mất cân bằng cực đoan (từ 1:100 đến 1:1000).
- Kiểm chứng chất lượng độ bất định (Calibration) thông qua hai thành phần: Epistemic Uncertainty (thiếu dữ liệu) và Aleatoric Uncertainty (nhiễu dữ liệu).
- Đánh giá khả năng tối ưu hóa cấu trúc mạng (Dynamic Sparse Training) sử dụng chiến lược cắt tỉa và tái tạo dựa trên độ bất định.
- Thử nghiệm trên các nhóm dữ liệu thực tế (Y tế, Công nghiệp, Tự nhiên).

---

## 2. Các Nhóm Benchmark Hỗ trợ

### Nhóm A: Dữ liệu Long-Tailed chuẩn (Controlled Long-Tailed Recognition)
- **CIFAR-100-LT**: Bộ dữ liệu ảnh tự nhiên thu nhỏ với tỷ lệ mất cân bằng được kiểm soát (e.g., 1:10, 1:50, 1:100). Dùng để kiểm chứng thuật toán cốt lõi và đo thời gian hội tụ.
- **ImageNet-LT**: Bộ dữ liệu ảnh tự nhiên quy mô lớn có phân bố đuôi dài.

### Nhóm B: Dữ liệu Bất thường / Lỗi Công nghiệp (Rare-Event / Anomaly Detection)
- **MVTec AD**: Phân loại các lỗi hiếm trong dây chuyền công nghiệp. Đặc điểm là mẫu lỗi rất ít so với mẫu bình thường.

### Nhóm C: Dữ liệu Y tế rủi ro cao (High-Stakes Case Study)
- **ISIC 2024**: Dữ liệu chẩn đoán ung thư hắc tố (melanoma) với tỷ lệ mất cân bằng cực đoan xấp xỉ 1:1000. Đòi hỏi độ nhạy (Sensitivity) cao và hạn chế tối đa chi phí dương tính giả (False Positive Cost).

---

## 3. Cấu trúc Pipeline & Môi trường

### Môi trường Yêu cầu (Prerequisites)
```bash
pip install torch torchvision numpy pandas scikit-learn matplotlib jupyter h5py wandb
```

### Các tệp tin cốt lõi
1. **`guds_edl_core.py`**: Chứa toàn bộ core logic (Backbone, Evidence Layer, Evidential Focal Loss, Adaptive Thresholds).
2. **`experiments/GUDS_EDL_Experiments.ipynb`**: Notebook tương tác, dùng để phân tích và chạy thí nghiệm trực tiếp.
3. **`experiments/run_benchmarks.bat`**: Script batch để cắm máy chạy quét qua nhiều tỷ lệ mất cân bằng khác nhau một cách tự động.

---

## 4. Quy trình Thực thi (Execution Workflow)

### Giai đoạn 1: Chuẩn bị Dữ liệu (Data Preparation)
- Khởi tạo DataLoader qua lớp `LongTailedDataset`.
- Áp dụng các kỹ thuật augmentation, tính toán trọng số lớp (Class Weights) để chuẩn bị cho Focal Loss.
- (Tùy chọn) Chuyển dữ liệu ảnh sang định dạng `HDF5` để tăng tốc độ I/O nếu sử dụng ổ cứng cơ.

### Giai đoạn 2: Cấu hình Mô hình (Model Initialization & Sparsification)
- Khởi tạo mạng nền (ví dụ: ResNet-18) và thay thế lớp phân loại Dense bằng `EvidenceLayer`.
- Gọi hàm `replace_conv2d_with_mdep(model)` để chuyển đổi toàn bộ các lớp tích chập thành cấu trúc thưa thớt 2:4 (NVIDIA 2:4 Structured Sparsity).

### Giai đoạn 3: Huấn luyện (Training)
- **Warmup (Dense Phase):** Các epoch đầu được huấn luyện với dạng Dense mask để tích lũy tín hiệu gradient ổn định.
- **Sparse Phase (Uncertainty-Guided Pruning & Regrowth):**
  - **Pruner (Microglia alias):** Cắt tỉa các liên kết gây nhiễu, dựa trên đạo hàm của tỷ lệ độ bất định dữ liệu (`u_a`).
  - **Regrower (Astrocyte alias):** Mọc lại các liên kết tại những khu vực có độ bất định tri thức cao (`u_e`), hướng sự tập trung của mạng vào các nhóm thiểu số chưa được biểu diễn tốt.
- Tính toán tổn thất bằng hàm `EvidentialFocalLoss` kết hợp với thuật toán bù trừ Asymmetric KL Divergence.

### Giai đoạn 4: Hiệu chỉnh và Đánh giá Thích ứng (Post-hoc Calibration & Adaptive Evaluation)
- Cố định trọng số mô hình và áp dụng **Temperature Scaling** trên tập Calibration để cân bằng độ tin cậy.
- Đánh giá trên 3 chế độ vận hành (Adaptive Operating Modes):
  1. **Balanced Utility**: Tối ưu hóa điểm chuẩn F1/Macro-AUROC.
  2. **High-Recall (Fail-Safe)**: Cố định Sensitivity $\ge 80\%$ (hoặc 95\%), các mẫu khó đoán (có `u_e` cao) sẽ tự động được gán cờ `Flagged for Human Review`.
  3. **Quality-Gated**: Sử dụng độ bất định dữ liệu (`u_a`) làm bộ lọc để tự động loại bỏ các ảnh quá mờ / nhiễu trước khi đưa ra quyết định.

---

## 5. Các Metrics Báo cáo Đầu ra
- **$\text{pAUC}_{0.80}$**: Điểm AUC đo trên khoảng Sensitivity $\ge 80\%$ (Rất quan trọng cho lĩnh vực rủi ro cao).
- **Macro-AUROC** và **Macro-F1**.
- **Global ECE** và **Minority ECE**: Sai số kỳ vọng độ tin cậy.
- **AURC (Area Under the Risk-Coverage Curve)**: Đánh giá khả năng dự đoán chọn lọc (Selective Classification).
- Các đồ thị lưu trong thư mục `artifacts/` (Reliability Diagrams, Risk-Coverage curves).
