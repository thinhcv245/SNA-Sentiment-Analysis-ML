import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

# Dữ liệu bình luận và nhãn tương ứng
binh_luan = [
    "@Kenichan I dived many times for the ball. Managed to save 50%  The rest go out of bounds",
    "my whole body feels itchy and like its on fire",
    "is upset that he can't update his Facebook by texting it... and might cry as a result  School today also. Blah!"
]

nhan = ["Tích cực", "Tiêu cực", "Tiêu cực"]  # Nhãn tương ứng

# Tính TF-IDF bằng TfidfVectorizer
vectorizer = TfidfVectorizer()
X_tfidf = vectorizer.fit_transform(binh_luan)

# Chuyển TF-IDF sang DataFrame
df_tfidf = pd.DataFrame(X_tfidf.toarray(), 
                        columns=vectorizer.get_feature_names_out(), 
                        index=["Bình luận 1", "Bình luận 2", "Bình luận 3"])

# Thêm cột Vector TF-IDF và Nhãn vào bảng kết quả
df_output = pd.DataFrame({
    'Bình luận': ["Bình luận 1", "Bình luận 2", "Bình luận 3"],
    'Vector TF-IDF': df_tfidf.apply(lambda row: row.values.tolist(), axis=1),
    'Nhãn': nhan
})

# Hiển thị kết quả
print(df_output)

# Lưu kết quả vào file CSV
df_output.to_csv("tfidf_with_labels.csv", index=False, encoding="utf-8")
