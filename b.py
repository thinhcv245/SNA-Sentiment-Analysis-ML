import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# 1. Đọc dữ liệu Sentiment140
url = 'sentiment140.csv'
columns = ['target', 'ids', 'date', 'query', 'user', 'text']
data = pd.read_csv(url, names=columns, encoding='latin-1')

# 2. Tiền xử lý dữ liệu văn bản
# Chuyển đổi thành chữ thường, loại bỏ dấu câu và URL
data['text'] = data['text'].str.lower()
data['text'] = data['text'].str.replace(r'http\S+', '', regex=True)
data['text'] = data['text'].str.replace(r'[^a-z\s]', '', regex=True)

# 3. Chuyển đổi cột 'date' thành định dạng datetime và tính toán thời gian kể từ tweet
data['date'] = data['date'].str.replace(r' [A-Z]{3} ', ' ')  # Loại bỏ múi giờ
data['date'] = pd.to_datetime(data['date'], errors='coerce')
first_tweet_time = data['date'].min()
data['time_since_post'] = (data['date'] - first_tweet_time).dt.total_seconds()

# 4. Tính tỷ lệ cảm xúc tích cực theo thời gian
time_bins = np.linspace(0, data['time_since_post'].max(), num=100)  # chia thành 100 khoảng thời gian
positive_sentiment = []
for t in time_bins:
    positive_ratio = len(data[(data['time_since_post'] <= t) & (data['target'] == 4)]) / len(data[data['time_since_post'] <= t])
    positive_sentiment.append(positive_ratio)

# 5. Áp dụng hàm decay (ví dụ: Rayleigh decay)
def rayleigh_decay(t, a=1.0, b=0.005):
    return a * np.exp(-b * t**2)

decay_values = rayleigh_decay(time_bins)

# 6. Vẽ đồ thị tỷ lệ cảm xúc tích cực và hàm decay
plt.figure(figsize=(10, 6))

# Vẽ tỷ lệ cảm xúc tích cực theo thời gian
plt.plot(time_bins, positive_sentiment, label='Positive Sentiment Proportion', color='green')

# Vẽ hàm Rayleigh decay
plt.plot(time_bins, decay_values, label='Rayleigh Decay Function', color='blue', linestyle='--')

# Thêm tiêu đề và nhãn
plt.title('Proportion of Positive Emotions Over Time (with Rayleigh Decay)')
plt.xlabel('Time (seconds since first tweet)')
plt.ylabel('Proportion of Positive Emotions')
plt.legend()
plt.grid(True)

# Hiển thị đồ thị
plt.show()
