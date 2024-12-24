import pandas as pd

def load_and_preprocess(data_path, processed_path):
    """
    Tải và xử lý dữ liệu thô từ Kaggle Sentiment140.
    """
    # Đọc dữ liệu thô
    columns = ["target", "id", "date", "flag", "user", "text"]
    df = pd.read_csv(data_path, encoding="latin-1", names=columns)

    # Chuyển cảm xúc thành tích cực (+1), trung lập (0), tiêu cực (-1)
    df["sentiment"] = df["target"].apply(lambda x: 1 if x == 4 else -1 if x == 0 else 0)

    # Chỉ giữ các cột cần thiết
    df = df[["user", "text", "sentiment"]]

    # Lưu dữ liệu đã xử lý
    df.to_csv(processed_path, index=False)
    return df
