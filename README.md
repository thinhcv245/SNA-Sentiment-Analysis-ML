# Phân Tích Cảm Xúc Trên Mạng Xã Hội

Dự án này thực hiện phân tích cảm xúc từ dữ liệu tweet bằng cách sử dụng các kỹ thuật học máy và xử lý ngôn ngữ tự nhiên. Dữ liệu được lấy từ tập dữ liệu Sentiment140, và các phương pháp như TF-IDF, đồ thị và mô phỏng lan truyền cảm xúc được áp dụng để phân tích.

## Nội Dung

- `src/`: Chứa mã nguồn chính của dự án.
  - `tf_analysis.py`: Phân tích tần suất từ (TF) của các bình luận.
  - `tf_idf_analysis.py`: Tính toán TF-IDF cho các bình luận.
  - `idf_analysis.py`: Tính toán IDF và tần suất xuất hiện của từ.
  - `sentiment140.py`: Xử lý và phân tích dữ liệu từ Sentiment140.
  - `preprocessing.py`: Tiền xử lý dữ liệu văn bản.
  - `run.py`: Chạy toàn bộ quy trình phân tích.
  - `tf_idf_in_comment.py`: Tính toán TF-IDF trong các bình luận.
  - `g.py`: Tạo và vẽ đồ thị từ dữ liệu.
  - `services.py`: Chứa các hàm tiện ích cho xử lý văn bản.
- `build.py`: Xây dựng đồ thị từ dữ liệu nodes và edges.
- `graph-tool.py`: Xác minh và vẽ đồ thị đã lưu.
- `requirements.txt`: Danh sách các thư viện cần thiết để chạy dự án.
- `.gitignore`: Các tệp và thư mục cần bỏ qua khi sử dụng Git.

## Cài Đặt

1. Clone repository về máy của bạn:
   ```bash
   git clone https://github.com/yourusername/yourrepository.git
   cd yourrepository
   ```

2. Cài đặt các thư viện cần thiết:
   ```bash
   pip install -r requirements.txt
   ```

## Sử Dụng

1. Đảm bảo rằng bạn đã có dữ liệu Sentiment140 trong thư mục `data/raw/`.
2. Chạy file `run.py` để bắt đầu phân tích:
   ```bash
   python src/run.py
   ```

3. Kết quả sẽ được lưu vào các file CSV và hình ảnh đồ thị sẽ được tạo ra trong thư mục hiện tại.

## Ghi Chú

- Đảm bảo rằng bạn đã cài đặt Python 3.x và pip.
- Nếu bạn gặp vấn đề với NLTK, hãy đảm bảo tải xuống các bộ dữ liệu cần thiết bằng cách mở Python và chạy:
  ```python
  import nltk
  nltk.download('stopwords')
  nltk.download('punkt')
  ```
