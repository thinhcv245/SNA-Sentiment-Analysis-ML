from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
import joblib
import os
import time
import threading
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score


#biến cục bộ
#vectorizer = TfidfVectorizer(max_features=5000)
vectorizer = TfidfVectorizer()

class Timer:
    def __init__(self):
        self.start_time = None
        self.running = False

    def start(self):
        """Bắt đầu hiển thị thời gian liên tục."""
        self.start_time = time.time()
        self.running = True
        threading.Thread(target=self._display_time, daemon=True).start()

    def _display_time(self):
        """Hiển thị thời gian đã chạy mỗi giây trên một dòng."""
        while self.running:
            elapsed_time = time.time() - self.start_time
            print(f"\rElapsed Time: {elapsed_time:.2f} seconds", end="", flush=True)  # Hiển thị trên cùng một dòng
            time.sleep(1)

    def stop(self):
        """Dừng hiển thị thời gian và trả về tổng thời gian."""
        self.running = False
        elapsed_time = time.time() - self.start_time
        print(f"\rTotal Time: {elapsed_time:.2f} seconds")  # In tổng thời gian
        return elapsed_time
    
def print_classification_report(y_true, y_pred, labels=None, target_names=None):
    """
    Hàm tùy chỉnh để hiển thị classification report chi tiết.

    Args:
        y_true (list or array): Nhãn thật.
        y_pred (list or array): Nhãn dự đoán.
        labels (list, optional): Danh sách các nhãn để báo cáo. Mặc định là None.
        target_names (list, optional): Danh sách tên các nhãn. Mặc định là None.

    Returns:
        None: Chỉ in ra báo cáo.
    """
    # Báo cáo tổng quát từ sklearn
    report = classification_report(y_true, y_pred, labels=labels, target_names=target_names)
    
    # Độ chính xác (accuracy)
    accuracy = accuracy_score(y_true, y_pred)
    
    print("\n===== Classification Report =====")
    print(report)
    print(f"Overall Accuracy: {accuracy:.2%}")
    print("=================================\n")
vectorizer = TfidfVectorizer()
from sklearn.metrics import classification_report, accuracy_score

def print_classification_report(y_true, y_pred, labels=None, target_names=None):
    """
    Hàm tùy chỉnh để hiển thị classification report chi tiết.

    Args:
        y_true (list or array): Nhãn thật.
        y_pred (list or array): Nhãn dự đoán.
        labels (list, optional): Danh sách các nhãn để báo cáo. Mặc định là None.
        target_names (list, optional): Danh sách tên các nhãn. Mặc định là None.

    Returns:
        None: Chỉ in ra báo cáo.
    """
    # Báo cáo tổng quát từ sklearn
    report = classification_report(y_true, y_pred, labels=labels, target_names=target_names)
    
    # Độ chính xác (accuracy)
    accuracy = accuracy_score(y_true, y_pred)
    
    print("\n===== Classification Report =====")
    print(report)
    print(f"Overall Accuracy: {accuracy:.2%}")
    print("=================================\n")

def load_model(dir_model_path, dir_vectorizer_path):
    # Tải mô hình và vectorizer từ tệp
    model_nb = joblib.load(dir_model_path)
    vectorizer = joblib.load(dir_vectorizer_path)
    print("Model and vectorizer have been loaded successfully.")
    return model_nb, vectorizer
def split_dataset(data):
    if 'clean_text' in data.columns:
        data['clean_text'] = data['clean_text'].fillna('')  # Thay thế NaN bằng chuỗi rỗng
    
    # Chuyển nhãn cảm xúc từ 0 và 4 thành 0 và 1
    data['target'] = data['target'].map({0: 0, 4: 1})  # Chuyển 4 thành 1

    # Dữ liệu đầu vào
    X = data['clean_text']  # Văn bản đã tiền xử lý
    y = data['target']  # Nhãn cảm xúc (0 hoặc 1)

    # Chuyển văn bản thành vector TF-IDF
    X_tfidf = vectorizer.fit_transform(X)

    # Chia tập dữ liệu
    X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

# Huấn luyện Naive Bayes
# Huấn luyện Naive Bayes và lưu mô hình
def training_native_bayes(data, model_dir):
    # Kiểm tra và xử lý các giá trị NaN trong cột clean_text
    # Huấn luyện Naive Bayes
    # Khởi tạo Timer
    timer = Timer()
    timer.start()

    X_train, X_test, y_train, y_test = split_dataset(data)
    print("Training Naive Bayes...")
    model_nb = MultinomialNB()
    model_nb.fit(X_train, y_train)
    timer.stop()
    # Dự đoán và đánh giá
    y_pred = model_nb.predict(X_test)
    print_classification_report(y_test, y_pred, labels=[0, 1], target_names=["Negative", "Positive"])

    # Kiểm tra và tạo thư mục lưu mô hình nếu chưa tồn tại
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    # Lưu mô hình và vectorizer vào tệp
    joblib.dump(model_nb, os.path.join(model_dir, 'naive_bayes_model.pkl'))  # Lưu mô hình Naive Bayes
    joblib.dump(vectorizer, os.path.join(model_dir, 'tfidf_vectorizer.pkl'))  # Lưu vectorizer

    print("Model and vectorizer have been saved successfully.")

    # Trả về mô hình và vectorizer
    return model_nb, vectorizer

#Support Vector Machine (SVM)
def training_SVM(data, model_dir):
    model_svm = SVC(kernel='linear', C=1.0)

    timer =Timer()

    # Bắt đầu đo thời gian
    timer.start()
    X_train, X_test, y_train, y_test = split_dataset(data)
    print("Training Support Vector Machine...")

    model_svm.fit(X_train, y_train)

    # Dự đoán và đánh giá
    y_pred_svm = model_svm.predict(X_test)
    # print(classification_report(y_test, y_pred_svm))
    print_classification_report(y_test, y_pred_svm, labels=[0, 1], target_names=["Negative", "Positive"])

    timer.stop()

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    
      # Save the model and vectorizer
    joblib.dump(model_svm, os.path.join(model_dir, 'svm_model.pkl'))  # Save the SVM model
    joblib.dump(vectorizer, os.path.join(model_dir, 'tfidf_vectorizer.pkl'))  # Save the vectorizer

    print(f"Model and vectorizer saved to {model_dir}")
    return model_svm