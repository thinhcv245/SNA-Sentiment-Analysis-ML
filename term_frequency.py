import pandas as pd
import math

def tinh_idf(comments, path, name_file):
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
    # Xuất ra file CSV
    tf_matrix.to_csv(path + name_file, index=True, encoding='utf-8')

def tinh_tf_idf(data, path, name_file):
    # Tạo danh sách từ
    words = data['Từ'].tolist()

    # Số lượng tài liệu (cột "Từ" không tính)
    N = len(data.columns) - 1  # Lấy số tài liệu từ số cột (trừ cột "Từ")

    # Tính df(t) và IDF(t)
    df = []
    idf = []

    # Lặp qua từng từ trong danh sách từ
    for word in words:
        # df(t) là số tài liệu chứa từ t (tính bằng cách đếm số cột có giá trị > 0 cho từ t)
        df_t = (data.iloc[:, 1:] > 0).sum(axis=0)  # Đếm số tài liệu có chứa từ
        df_t_value = (df_t > 0).sum()  # Đếm số tài liệu có chứa từ (nếu > 0)

        # In ra df(t) để kiểm tra
        print(f"df({word}): {df_t_value}")

        # IDF(t) = log(N / df(t)) với điều kiện df(t) > 0
        if df_t_value > 0:
            idf_t = math.log(N / df_t_value)
        else:
            idf_t = 0  # Nếu df(t) = 0 thì IDF(t) không xác định, cho là 0
        
        df.append(df_t_value)
        idf.append(idf_t)

    # Tạo bảng IDF
    idf_df = pd.DataFrame({
        'Từ': words,
        'df(t)': df,
        'IDF(t)': idf
    })

    # Lưu kết quả ra file CSV
    idf_df.to_csv(path + name_file, encoding='utf-8', index=False)
    print(idf_df)


# Dữ liệu bình luận ví dụ
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

# Đường dẫn và tên file xuất
path = "matrix/"
name_file = "tf_matrix.csv"

# Tính IDF
# tinh_idf(comments, path, name_file)

# Đọc lại dữ liệu từ file đã lưu
data = pd.read_csv(path + name_file)

# Tính TF-IDF
name_file2 = 'tf_idf_results.csv'
tinh_tf_idf(data, path, name_file2)
