import pandas as pd
import os
# Đọc dataset Sentiment140
def check_sentiment(path_data):
    # Đọc dữ liệu mà không có header
    data = pd.read_csv(path_data, encoding='latin-1', header=None)

    # Tạo tên cột sau khi đọc dữ liệu
    data.columns = ['target', 'ids', 'date', 'query', 'user', 'text']

    # Hiển thị 5 dòng đầu tiên
    print(data.head())
def getdata(path_data):
    # Đọc dữ liệu mà không có header
    data = pd.read_csv(path_data, encoding='latin-1', header=None)
    data.columns = ['target', 'ids', 'date', 'query', 'user', 'text']
    return data
def get400K_csv(data, output_path):
    # lấy dữ liệu 200k của taget 0 và 200k của taget 4
    df_target_0 = data[data['target'] == 0]
    df_target_4 = data[data['target'] == 4]

    # Lấy 200.000 dòng từ mỗi nhóm
    df_target_0_sample = df_target_0.sample(n=200000, random_state=42)
    df_target_4_sample = df_target_4.sample(n=200000, random_state=42)

    # Kết hợp lại để tạo DataFrame mới
    df_sample = pd.concat([df_target_0_sample, df_target_4_sample])

    # Đảm bảo rằng dữ liệu đã được trộn (shuffle)
    df_sample = df_sample.sample(frac=1, random_state=42).reset_index(drop=True)

     # Kiểm tra xem file đã tồn tại chưa
    if not os.path.exists(output_path):
        # Nếu không tồn tại, tạo mới và lưu
        df_sample.to_csv(output_path, index=False)
        print(f"File đã được tạo và lưu tại {output_path}")
    else:
        print(f"File {output_path} đã tồn tại.")

    # Hiển thị một số dòng đầu tiên của dữ liệu đã lấy mẫu
    print(df_sample.head())


#main
if __name__ == "__main__":
    # Kiểm tra sentiment của từng dòng dữ liệu
    path_data = "sentiment140/training.1600000.processed.noemoticon.csv"
    path_data_output = "sentiment140/training.400000.csv"
    check_sentiment(path_data)
    data = getdata(path_data)
    get400K_csv(data, path_data_output)