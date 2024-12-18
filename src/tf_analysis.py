import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

# Dữ liệu: Ba bình luận
binh_luan_1 = "@Kenichan I dived many times for the ball. Managed to save 50%  The rest go out of bounds"
binh_luan_2 = "my whole body feels itchy and like its on fire"
binh_luan_3 = "is upset that he can't update his Facebook by texting it... and might cry as a result  School today also. Blah!"

# Tập hợp các bình luận vào danh sách
corpus = [binh_luan_1, binh_luan_2, binh_luan_3]

# Tạo CountVectorizer để tính TF (Term Frequency)
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(corpus)

# Tạo DataFrame để hiển thị kết quả
df_tf = pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names_out(), 
                     index=['TF trong Bình luận 1', 'TF trong Bình luận 2', 'TF trong Bình luận 3'])

# Chuyển cột thành hàng để có dạng như yêu cầu
df_tf = df_tf.T.reset_index()
df_tf.columns = ['Từ', 'TF trong Bình luận 1', 'TF trong Bình luận 2', 'TF trong Bình luận 3']

# Hiển thị kết quả
print(df_tf)

# Lưu kết quả vào file CSV
df_tf.to_csv('tf_table.csv', index=False, encoding='utf-8')
