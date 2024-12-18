import pandas as pd
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.corpus import stopwords
import os

# Download stopwords nếu chưa có
# nltk.download('stopwords')

# Bước 1: Load dữ liệu từ Sentiment140 CSV
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
df = pd.read_csv(os.path.join(base_dir, 'data', 'raw', 'sentiment140.csv'), encoding='ISO-8859-1', header=None)
df.columns = ['target', 'id', 'date', 'flag', 'user', 'text']
df = df.sample(20)

# Chọn một tập nhỏ dữ liệu để thử nghiệm (ví dụ: 1000 tweet đầu tiên)
texts = df['text']

# Bước 2: Tiền xử lý văn bản
stop_words = set(stopwords.words('english'))  # Stopwords tiếng Anh

# Hàm tiền xử lý văn bản
def preprocess(text):
    text = re.sub(r'http\S+', '', text)  # Xóa URL
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Xóa ký tự không phải chữ cái
    text = text.lower()  # Chuyển thành chữ thường
    words = text.split()
    words = [w for w in words if w not in stop_words]  # Loại bỏ stopwords
    return ' '.join(words)

# Tiền xử lý toàn bộ dữ liệu
cleaned_texts = [preprocess(t) for t in texts]

# Bước 3: Tính tần suất df(t) và IDF(t)
vectorizer = TfidfVectorizer(use_idf=True, smooth_idf=False)  # Tính TF-IDF
X = vectorizer.fit_transform(cleaned_texts)

# Lấy danh sách từ và các chỉ số IDF
vocab = vectorizer.get_feature_names_out()
idf_scores = vectorizer.idf_

# Tính df(t) từ IDF
N = len(cleaned_texts)  # Tổng số văn bản
df_scores = N / np.exp(idf_scores)  # df(t) = N / exp(IDF(t))

# Bước 4: Tạo bảng kết quả df(t) và IDF(t)
df_idf = pd.DataFrame({
    'Từ': vocab,
    'df(t)': np.round(df_scores, 0).astype(int),  # Làm tròn số và chuyển sang int
    'IDF(t)': np.round(idf_scores, 2)  # Làm tròn IDF đến 2 chữ số
})

# Sắp xếp kết quả theo df(t)
df_idf = df_idf.sort_values(by='df(t)', ascending=False).reset_index(drop=True)

# Bước 5: Hiển thị kết quả
print(df_idf)

# Lưu kết quả ra file CSV
df_idf.to_csv('idf_results.csv', index=False, encoding='utf-8')
