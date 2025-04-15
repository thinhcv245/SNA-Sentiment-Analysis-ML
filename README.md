# Dự Án Phân Tích Cảm Xúc Trên Mạng Xã Hội - Mô hình Depression Influence Cascade (DepIC)

## Mô Tả
Dự án này sử dụng mô hình lan truyền cảm xúc để phân tích và trực quan hóa cảm xúc của người dùng trên mạng xã hội. Dữ liệu được lấy từ Kaggle Sentiment140 và được xử lý để xây dựng đồ thị mạng xã hội. Các mô hình lan truyền cảm xúc được áp dụng để đánh giá ảnh hưởng của các tham số khác nhau đến cảm xúc tích cực.

## Công Nghệ Sử Dụng
- Python
- NetworkX
- NumPy
- Matplotlib
- Pandas
- Scikit-learn

## Cài Đặt
1. Clone repository này về máy của bạn:
   ```bash
   git clone https://github.com/yourusername/yourrepository.git
   cd yourrepository
   ```

2. Cài đặt các thư viện cần thiết:
   ```bash
   pip install -r requirements.txt
   ```

## Cấu Hình
- Tạo file `config.py` với các biến sau:
  ```python
  DATA_PATH = "./data/sentiment140.csv"
  PROCESSED_PATH = "./data/processed.csv"
  ALPHA = 0.2
  BETA_POS = 0.3
  BETA_NEG = 0.2
  TIME_DECAY = 0.003
  MAX_STEPS = 80
  ```

## Mô Hình DepIC
Mô hình DepIC được sử dụng để lan truyền cảm xúc trong mạng xã hội. Cảm xúc của mỗi nút được cập nhật theo công thức sau:

$$ O_v(t + 1) = O_v(t) + \Delta O_v $$

Giới hạn cảm xúc được đảm bảo như sau:

$$ O_v(t + 1) = \text{min}(1, \text{max}(-1, O_v(t + 1))) $$

Khi số bước (Step) nhỏ hơn 10, có sự biến động mạnh, đặc biệt khi \( \alpha = 0.2 \), cho thấy mô hình đang chưa ổn định. Khi số bước lớn hơn 20, cảm xúc dần về hội tụ nhưng tỷ lệ cảm xúc phụ thuộc vào giá trị của \( \alpha \).

## 🧠 Thuật toán (Pseudocode)

```plaintext
Input:
    - G(U, E): đồ thị mạng xã hội
    - T: số bước lan truyền tối đa
    - Y₀: cảm xúc ban đầu của các nút
    - α, β, θ: tham số mô hình

Output:
    - Y_T: cảm xúc cuối cùng sau T bước hoặc khi hội tụ

Thuật toán:
1. Khởi tạo Y ← Y₀

2. Lặp với mỗi bước thời gian t từ 1 đến T:
    - Với mỗi nút u trong U:
        - Với mỗi nút v ∈ Neighbors(u):
            - Tính: ΔO_v = α × β × (O_u − O_v) × f(t; θ)
            - Cập nhật: O_v ← O_v + ΔO_v
            - Giới hạn O_v trong [−1, 1]

    - Nếu không có thay đổi nào trong O_v của tất cả v → dừng sớm

3. Trả về Y_T (tập cảm xúc cuối cùng)
```

## Sử Dụng
1. Tải dữ liệu từ Kaggle Sentiment140 và lưu vào thư mục `data/`.
2. Chạy file `main.py` để bắt đầu quá trình xử lý và phân tích:
   ```bash
   python main.py
   ```

3. Các file khác trong dự án:
   - `degree.py`: Trực quan hóa phân phối bậc của đồ thị.
   - `decay.py`: Vẽ biểu đồ ảnh hưởng của các hàm suy giảm thời gian đến cảm xúc.
   - `alpha.py`: Vẽ biểu đồ ảnh hưởng của tham số alpha đến tỷ lệ cảm xúc tích cực.
   - `compare.py`: So sánh các mô hình lan truyền cảm xúc khác nhau.
  
## Kết Quả
Dưới đây là ví dụ về kết quả từ file `alpha.py`:
![alpha](https://github.com/user-attachments/assets/fe87418e-eb61-4c2a-b308-74f55920cfc2)

## Ghi Chú
- Đảm bảo rằng bạn đã cài đặt Python 3.x và pip.
- Dữ liệu đầu vào cần phải được định dạng đúng theo yêu cầu của dự án.

## Liên Hệ
Nếu bạn có bất kỳ câu hỏi nào, vui lòng liên hệ với tôi qua email: thinh.chauvan2405@gmail.com
