import pandas as pd

# Đọc dữ liệu từ file CSV
data = pd.read_csv("sentiment140/elon_musk.csv")

# Thay đổi nhãn trong cột 'target' theo yêu cầu
sentiment_map = {'negative': 0, 'neutral': 2, 'positive': 4}
data['target'] = data['target'].map(sentiment_map)

# Kiểm tra lại kết quả
print(data.head())

# Lưu lại dữ liệu đã cập nhật vào file mới (nếu cần)
data.to_csv("sentiment140/elon_musk.csv", index=False)