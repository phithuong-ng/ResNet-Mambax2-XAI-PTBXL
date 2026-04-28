🩺 RMCN: ResNet-Mamba Cascaded Network for Multi-label ECG DiagnosisDự án nghiên cứu: Xây dựng mạng nối tầng ResNet-Mamba trong chẩn đoán đa nhãn bệnh lý tim mạch với khả năng giải thích bằng Grad-CAM/SHAP.📌 1. Giới thiệu tổng quan (Overview)Dự án này tập trung giải quyết bài toán chẩn đoán đa nhãn (Multi-label) trên dữ liệu ECG 12 đạo trình. Điểm đột phá của hệ thống là sự kết hợp giữa khả năng trích xuất đặc trưng hình thái của ResNet1D và khả năng mô hình hóa chuỗi dài với độ phức tạp tuyến tính $O(N)$ của Mamba (Selective State Space Model).✨ Tính năng nổi bật:Hybrid Architecture: Kết hợp ResNet (Local Perception) và Mamba (Global Context).Clinical Integrity: Quy trình tiền xử lý bảo toàn biên độ vật lý (mV) để chẩn đoán phì đại tâm thất (HYP).XAI Integrated: Minh bạch hóa mô hình bằng bản đồ nhiệt Grad-CAM và SHAP.Edge-AI Ready: Tối ưu hóa hiệu suất tính toán cho các thiết bị nhúng tại Lab EDABK.🛠 2. Kiến trúc hệ thống (Technical Architecture)🛰 Quy trình tiền xử lý (Preprocessing)Chúng tôi áp dụng các tiêu chuẩn y sinh khắt khe nhất:Butterworth Bandpass Filter (3rd Order): Dải thông $[0.5Hz, 45.0Hz]$ để loại bỏ nhiễu trôi đường nền và nhiễu cơ.Zero-phase Filtering: Sử dụng thuật toán lọc hai chiều (Forward-Backward) để triệt tiêu hoàn toàn độ lệch pha $\Delta \phi$, giữ các đỉnh sóng P-QRS-T đúng vị trí lâm sàng.Physical Amplitude Preservation: Tuyệt đối KHÔNG chuẩn hóa (Normalization) để bảo vệ giá trị điện thế (Volt) phục vụ chẩn đoán HYP.🧠 Mô hình RMCN (ResNet-Mamba Cascaded Network)Mô hình thực hiện phép tính toán học thông qua các khối:ResNet Backbone: Học hàm phần dư $y = \mathcal{F}(x, \{W_i\}) + x$.Mamba Block (S6): Sử dụng cơ chế quét chọn lọc (Selective Scan) dựa trên tham số bước thời gian $\Delta$ để nén ngữ cảnh toàn cục vào trạng thái ẩn $h(t)$.📂 3. Cấu trúc thư mục (Project Structure)Tuân thủ nghiêm ngặt yêu cầu nộp bài của giảng viên:Plaintext.
├── 01 Slide/          # Slide báo cáo PDF/PPTX
├── 02 Report/         # File báo cáo chuẩn IEEE (.docx)
├── 03 Code/           # Toàn bộ mã nguồn dự án
│   ├── models/        # Định nghĩa ResNet & Mamba blocks
│   ├── utils/         # Script lọc tín hiệu & tiền xử lý
│   ├── train.py       # Script huấn luyện mô hình
│   └── explain.py     # Script tạo bản đồ nhiệt Grad-CAM
├── 04 Datasets/       # Hướng dẫn tải PTB-XL
└── 05 References/     # Các bài báo tham khảo Q1/Q2
🚀 4. Hướng dẫn cài đặt & Chạy (Installation & Usage)📦 Yêu cầu hệ thốngPython >= 3.9CUDA (Nếu chạy bằng GPU)Thư viện: torch, mamba-ssm, scipy, pandas, pytorch-grad-cam.🔨 Cài đặt nhanhBash# Clone dự án
git clone https://github.com/nguyenphithuong/ECG-ResNet-Mamba.git
cd ECG-ResNet-Mamba/03_Code

# Cài đặt thư viện
pip install -r requirements.txt
🏃 Huấn luyện mô hìnhBashpython train.py --data_path ../04_Datasets/ptbxl/ --epochs 50 --batch_size 64
📊 5. Kết quả thực nghiệm (Experimental Results)Dưới đây là bảng so sánh hiệu năng trên tập Test (Fold 10):ModelMacro F1Macro AUCGhi chúResNet1D (Raw)0.72970.9426BaselineRMCN (Ours)0.73800.9500Bảo toàn mV🔍 Giải thích mô hình (XAI)Bản đồ nhiệt Grad-CAM chỉ ra rằng mô hình tập trung chính xác vào:MI/STTC: Đoạn ST-T và phức bộ QRS.HYP: Biên độ của đỉnh sóng R (chỉ có được nhờ việc không Normalize).
