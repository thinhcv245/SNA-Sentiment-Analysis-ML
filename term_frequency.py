import pandas as pd

# # Danh sách các bình luận
# data =  pd.read_csv('sentiment140/processed_training.1600000_2.csv')
# # Lấy 10 bình luận đầu tiên
# comments = data['clean_text'][:10].astype(str)
# filtered_comments = comments[comments.apply(lambda x: len(x.split()) <= 5)]
# print("10 đoạn bình luận đầu tiên:")
# print(comments)
comments = [
    "This is a great day", 
    "I love programming", 
    "Python is awesome", 
    "Data science is fascinating", 
    "Machine learning is the future", 
    "I enjoy learning new things", 
    "This is an example comment", 
    "The weather is nice today", 
    "I love reading books", 
    "Life is beautiful"
]

# Tạo danh sách tất cả từ duy nhất
words = list(set(" ".join(comments).replace('.', '').split()))

# Tạo cột động dựa trên số lượng bình luận
column_names = [f'TF-BL {i+1}' for i in range(len(comments))]

# Khởi tạo bảng TF với 0
tf_matrix = pd.DataFrame(0, index=words, columns=column_names)

# Đếm tần suất từ trong từng bình luận
for i, comment in enumerate(comments):
    for word in comment.replace('.', '').split():
        tf_matrix.loc[word, f'TF-BL {i+1}'] += 1

# Sắp xếp lại từ theo thứ tự bảng chữ cái
tf_matrix = tf_matrix.sort_index()
tf_matrix.index.name = 'Từ'

# Hiển thị kết quả
print("\nBảng TF:")
print(tf_matrix)
# Xuất ra file text
tf_matrix.to_csv('tf_matrix.csv', encoding='utf-8')
