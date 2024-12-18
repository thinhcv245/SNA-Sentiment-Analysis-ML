import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

# Danh sách các bình luận
binh_luan = [
    "@Kenichan I dived many times for the ball. Managed to save 50%  The rest go out of bounds",
    "my whole body feels itchy and like its on fire",
    "is upset that he can't update his Facebook by texting it... and might cry as a result  School today also. Blah!"
]

# Tính TF-IDF
vectorizer = TfidfVectorizer()
X_tfidf = vectorizer.fit_transform(binh_luan)

# Chuyển kết quả thành DataFrame
df_tfidf = pd.DataFrame(X_tfidf.toarray(), columns=vectorizer.get_feature_names_out(), index=['TF-IDF trong Bình luận 1', 'TF-IDF trong Bình luận 2', 'TF-IDF trong Bình luận 3'])

df_tfidf = df_tfidf.T.reset_index()
df_tfidf.columns = ['Từ', 'TF-IDF trong Bình luận 1', 'TF-IDF trong Bình luận 2', 'TF-IDF trong Bình luận 3']

print(df_tfidf)

df_tfidf.to_csv('tfidf_table.csv', index=False, encoding='utf-8')