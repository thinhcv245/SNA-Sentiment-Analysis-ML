import pandas as pd

# Đọc file CSV
df = pd.read_csv('d.csv')

# Lấy cột đầu tiên (giả sử cột đầu tiên có tên là 'id_column')
ids = df.iloc[:, 0].tolist()

# Chuyển danh sách thành chuỗi dạng IN (...)
ids_in_clause = "IN (" + ",".join(map(str, ids)) + ")"

# In kết quả
print(ids_in_clause)
