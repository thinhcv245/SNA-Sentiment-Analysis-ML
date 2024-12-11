from lib import load_data, clean_text, save_data
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

def print_data(path):
    data = load_data(path)
    print(data.head())  # Display the first few rows of the dataset
def fit_transform_data(path_data):
    data_clean = load_data(path_output)
    if data_clean is not None:
        # Kiểm tra và xử lý các giá trị NaN trong cột clean_text
        if 'clean_text' in data_clean.columns:
            data_clean['clean_text'] = data_clean['clean_text'].fillna('')  # Thay thế NaN bằng chuỗi rỗng
            
            # Chuyển văn bản thành vector số sử dụng TF-IDF
            tfidf = TfidfVectorizer(max_features=5000)  # Chỉ giữ lại 5000 từ phổ biến nhất
            X = tfidf.fit_transform(data_clean['clean_text'])
            # Hiển thị kích thước của ma trận TF-IDF
            print(f"TF-IDF matrix size: {X.shape}")
            return X
            
        else:
            print("Column 'clean_text' not found in the dataset.")
    else:
        print("Failed to load the data.")
def train_test_split_data(x,dataset,strings=None):
    X_train, X_test, y_train, y_test = train_test_split(X, dataset[strings], test_size=0.2, random_state=42)
    return X_test, X_train, y_train, y_test

# Main code execution
if __name__ == '__main__':
    # Load data
    path_output = "sentiment140/processed_training.400000.csv"
    data = load_data(path_output)
    X = fit_transform_data(path_output)
    X_test, X_train, y_train, y_test = train_test_split_data(X, data, strings='target')
    print(X_train.shape, X_test.shape)
