# 🩺 RMCN: ResNet-Mamba Cascaded Network for Multi-label ECG Diagnosis

![HUST](https://img.shields.io/badge/University-Hanoi%20University%20of%20Science%20and%20Technology-red)
![Lab](https://img.shields.io/badge/Lab-EDABK-blue)
![Python](https://img.shields.io/badge/Python-3.9+-green?logo=python)
![PyTorch](https://img.shields.io/badge/Framework-PyTorch%202.0+-ee4c2c?logo=pytorch)
![License](https://img.shields.io/badge/License-MIT-yellow)

> **Dự án nghiên cứu:** Xây dựng mạng nối tầng ResNet-Mamba trong chẩn đoán đa nhãn bệnh lý tim mạch với khả năng giải thích bằng Grad-CAM/SHAP. Nghiên cứu tập trung giải quyết bài toán "hộp đen" trong Y tế Số và tối ưu hóa hiệu suất tính toán $O(N)$ trên chuỗi thời gian dài.

---

## 📌 1. Giới thiệu tổng quan (Overview)

Dự án này tập trung giải quyết bài toán chẩn đoán đa nhãn (Multi-label) trên dữ liệu ECG 12 đạo trình. Điểm đột phá của hệ thống là sự kết hợp giữa khả năng trích xuất đặc trưng hình thái của **ResNet1D** và khả năng mô hình hóa chuỗi dài với độ phức tạp tuyến tính $O(N)$ của **Mamba (Selective State Space Model)**.

### ✨ Tính năng nổi bật:
- **Hybrid Architecture:** Kết hợp ResNet (Local Perception) để quét vi mô hình thái sóng và Mamba (Global Context) để liên kết nhịp điệu toàn cục.
- **Clinical Integrity:** Quy trình tiền xử lý tuyệt đối bảo toàn biên độ vật lý (mV) để phục vụ chẩn đoán bệnh lý thay đổi điện thế như Phì đại tâm thất (HYP).
- **XAI Integrated:** Minh bạch hóa mô hình bằng bản đồ nhiệt Grad-CAM và SHAP, ánh xạ trực tiếp vùng chẩn đoán lên sóng thô (Raw waveform).
- **Edge-AI Ready:** Tối ưu hóa hiệu suất tính toán cho các thiết bị nhúng và FPGA tại Lab EDABK.

---

## 🛠 2. Kiến trúc hệ thống (Technical Architecture)

### 🛰 Quy trình tiền xử lý (Preprocessing)
Chúng tôi áp dụng các tiêu chuẩn y sinh khắt khe nhất để làm sạch dữ liệu PTB-XL:
1. **Butterworth Bandpass Filter (3rd Order):** Dải thông $[0.5Hz, 45.0Hz]$ để loại bỏ nhiễu trôi đường nền (Baseline wander) do hô hấp và nhiễu cơ/điện lưới.
2. **Zero-phase Filtering (filtfilt):** Sử dụng thuật toán lọc hai chiều (Forward-Backward) để triệt tiêu hoàn toàn độ lệch pha $\Delta \phi$, đảm bảo các đỉnh sóng P-QRS-T nằm đúng vị trí thời gian thực tế:
   $$\angle H_{total}(e^{j\omega}) = \angle H(e^{j\omega}) + \angle H(e^{-j\omega}) = 0$$
3. **Physical Amplitude Preservation:** Giữ nguyên tín hiệu ở hệ quy chiếu milli-Volt (mV), loại bỏ hoàn toàn các kỹ thuật chuẩn hóa (Z-score/Min-Max) vốn làm mất dữ liệu chẩn đoán lâm sàng.

### 🧠 Mô hình RMCN (ResNet-Mamba Cascaded Network)
Hệ thống học sâu được thiết kế theo dạng nối tầng:
- **ResNet Backbone:** Học hàm phần dư $y = \mathcal{F}(x, \{W_i\}) + x$ qua các lớp Conv1D để nhận diện sự biến dạng của phức bộ QRS và đoạn ST.
- **Mamba Block (S6):** Sử dụng cơ chế quét chọn lọc (Selective Scan) dựa trên tham số bước thời gian $\Delta$, nén ngữ cảnh chuỗi dài 10 giây vào trạng thái ẩn $h(t)$ thông qua phương trình vi phân liên tục được rời rạc hóa:
   $$h'(t) = Ah(t) + Bx(t)$$

---

## 📂 3. Cấu trúc thư mục (Project Structure)

Tuân thủ nghiêm ngặt chuẩn mực tổ chức mã nguồn nghiên cứu:

```text
.
├── 01_Slide/          # Slide báo cáo PDF/PPTX
├── 02_Report/         # File báo cáo chuẩn IEEE (.docx)
├── 03_Code/           # Toàn bộ mã nguồn dự án
│   ├── models/        # Định nghĩa ResNet1D & Mamba blocks
│   ├── utils/         # Script lọc tín hiệu (Butterworth, filtfilt)
│   ├── train.py       # Script huấn luyện mô hình (AdamW, BCEWithLogitsLoss)
│   └── explain.py     # Script tạo bản đồ nhiệt Grad-CAM
├── 04_Datasets/       # Dữ liệu PTB-XL & Script tải dữ liệu tự động
└── 05_References/     # Các bài báo tham khảo Q1/Q2 (PDFs)
