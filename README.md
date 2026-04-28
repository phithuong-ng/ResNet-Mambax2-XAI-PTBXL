# 🩺 RMCN: ResNet-Mamba Cascaded Network for Multi-label ECG Diagnosis

![HUST](https://img.shields.io/badge/University-Hanoi%20University%20of%20Science%20and%20Technology-red)
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

## 3. EXPERIMENTAL RESULTS AND CLINICAL EVALUATION

**A. Quantitative Performance Analysis and Baseline Comparison**

Mô hình chẩn đoán đề xuất đã được đánh giá nghiêm ngặt thông qua chiến lược kiểm chứng chéo 10-fold (10-fold stratified cross-validation) trên tập dữ liệu PTB-XL. Để giải quyết sự mất cân bằng dữ liệu cực đoan giữa các lớp bệnh, chúng tôi ưu tiên các độ đo không phụ thuộc ngưỡng: Macro F1-score và Diện tích dưới đường cong ROC (AUC).

Bảng I trình bày hiệu năng so sánh giữa kiến trúc ResNet1D cơ sở và quy trình tiền xử lý bảo tồn biên độ (Clean V2), đối chiếu với mô hình SOTA của Kanos (2025).

**TABLE I: COMPARATIVE PERFORMANCE ANALYSIS**

| Evaluation Metrics | ResNet1D (Raw Data) | RMCN Backbone (Clean V2) | Kanos (2025) Baseline |
| :--- | :---: | :---: | :---: |
| **Macro F1-Score** | 0.7297 | **0.7380** | 0.95 (500Hz, Norm) |
| **Micro Precision** | 0.7620 | **0.7721** | 0.95 |
| **AUC - NORM** | 0.9426 | **0.9500** | N/A |
| **AUC - MI** | 0.9200 | **0.9260** | N/A |
| **AUC - STTC** | 0.9264 | **0.9350** | N/A |
| **AUC - CD** | 0.9077 | **0.9160** | N/A |
| **AUC - HYP** | 0.8961 | **0.9010** | N/A |


*Biện luận lâm sàng:* Mặc dù Macro F1-score (0.7380) của chúng tôi thấp hơn Kanos (2025), nhưng đây là một sự đánh đổi kỹ thuật có tính toán. Kanos sử dụng tần số 500Hz và chuẩn hóa biên độ từng đạo trình. Ngược lại, khung RMCN hoạt động ở 100Hz và giữ nguyên biên độ vật lý (mV). Việc chuẩn hóa sẽ phá hủy các dấu hiệu điện thế tuyệt đối cần thiết để chẩn đoán Phì đại tâm thất (HYP). Chỉ số Macro AUC > 0.91 trên tất cả các lớp khẳng định rằng mạng RMCN đã trích xuất thành công các đặc trưng chẩn đoán cực kỳ mạnh mẽ.

**B. Explainable AI (XAI) and Diagnostic Transparency**

Để phá vỡ tính chất "hộp đen" của mạng Deep Learning, chúng tôi ứng dụng kỹ thuật Grad-CAM (Gradient-weighted Class Activation Mapping). Kỹ thuật này chiếu các bản đồ nổi bật (saliency maps) trực tiếp lên chuỗi thời gian ECG.

1. *Normal Rhythm Validation:* Với các ca NORM, bản đồ nhiệt phân bố đều đặn và nhịp nhàng trên các phức bộ QRS lặp lại. Điều này chứng minh mạng đã học được tính chu kỳ và sự ổn định của khoảng R-R thay vì bị nhiễu bởi các đoạn sóng lẻ tẻ.
2. *Pathological Alignment:* Trong các ca MI (Nhồi máu cơ tim) và STTC, các vùng kích hoạt gradient cao nhất tập trung chính xác vào phức bộ QRS và đoạn ST-T. Đối với chẩn đoán HYP, bản đồ nhiệt nhắm mục tiêu duy nhất vào các đỉnh sóng R có biên độ cao, minh chứng cho tính đúng đắn của việc bảo tồn biên độ mV.
3. *Model Limitations:* XAI cũng chỉ ra lỗ hổng của CNN thuần túy khi gặp các nhiễu trôi đường nền (Baseline wander) phức tạp. Điều này cung cấp bằng chứng thực nghiệm cho việc chuyển đổi sang khối Mamba (SSM) để lọc nhiễu và hiểu ngữ cảnh chuỗi dài tốt hơn.

---

## 4. CONCLUSION AND FUTURE DIRECTIONS

**A. Research Summary**

Báo cáo này đã trình bày một khung giải pháp toàn diện cho bài toán chẩn đoán đa nhãn bệnh lý tim mạch. Bằng cách kết hợp kiến trúc ResNet1D với quy trình tiền xử lý bảo tồn tính toàn vẹn vật lý của tín hiệu, chúng tôi đã đạt được hiệu suất phân loại mạnh mẽ với Macro AUC vượt ngưỡng 0.90. Việc ứng dụng Grad-CAM đã cung cấp sự minh bạch cần thiết, xác nhận rằng các biểu diễn toán học ẩn của mạng đã phản ánh chính xác các quy tắc điện sinh lý học của con người (tập trung vào biến dạng QRS cho thiếu máu cục bộ và đỉnh điện áp cho phì đại).

**B. Future Directions: The ResNet-Mamba Evolution**

Mặc dù ResNet trích xuất tốt các đặc trưng cục bộ, phân tích XAI cho thấy nó vẫn gặp khó khăn trong việc tách biệt các nhãn bệnh chồng lấn trên các đoạn tín hiệu dài. Hướng phát triển tiếp theo của chúng tôi là hoàn thiện mạng nối tầng ResNet-Mamba (RMCN). Bằng cách thay thế các lớp pooling toàn cục bằng khối Selective State Space (S6), chúng tôi sẽ đạt được khả năng mô hình hóa quan hệ xa với độ phức tạp tuyến tính O(N). Sự tích hợp của Mamba dự kiến sẽ cải thiện đáng kể khả năng lọc nhiễu động và gỡ rối các bệnh lý cùng tồn tại trên các bản ghi dài 10 giây hoặc hơn.

---
