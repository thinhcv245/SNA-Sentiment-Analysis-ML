import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

def show_coufusion_matrix(model_dir, path_model,path_name):
    # Tải mô hình và vectorizer đã lưu
    model = joblib.load(os.path.join(model_dir, path_model))
    tfidf_vectorizer = joblib.load(os.path.join(model_dir, path_name))

    data = pd.read_csv("sentiment140/processed_training.1600000_2.csv")

    if 'clean_text' in data.columns:
        data['clean_text'] = data['clean_text'].fillna('')  # Thay thế NaN bằng chuỗi rỗng

    data['target'] = data['target'].map({0: 0, 4: 1})  # Chuyển 4 thành 1

    X_raw = data['clean_text']  # Văn bản đã tiền xử lý
    y = data['target']  # Nhãn cảm xúc (0 hoặc 1)

    X_tfidf = tfidf_vectorizer.transform(X_raw)  # Chuyển đổi dữ liệu thô thành ma trận TF-IDF

    X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.2, random_state=42)

    y_pred = model.predict(X_test)

    # Hiển thị báo cáo đánh giá
    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["Negative", "Positive"]))

    # Vẽ confusion matrix
    conf_matrix = confusion_matrix(y_test, y_pred, labels=[0, 1])
    disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=["Negative", "Positive"])
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.show()
def show_coufusion_matrix_svm(model_dir, path_model, path_vectorizer, path_pca):
    # Tải mô hình và vectorizer đã lưu
  # Tải mô hình và các đối tượng đã lưu
    model_svm = joblib.load(os.path.join(model_dir, path_model))
    tfidf_vectorizer = joblib.load(os.path.join(model_dir, path_vectorizer))
    pca = joblib.load(os.path.join(model_dir, path_pca))

    # Tải dữ liệu
    data = pd.read_csv("sentiment140/processed_training.1600000_2.csv")

    if 'clean_text' in data.columns:
        data['clean_text'] = data['clean_text'].fillna('')  # Thay thế NaN bằng chuỗi rỗng

    data['target'] = data['target'].map({0: 0, 4: 1})  # Chuyển 4 thành 1

    # Tách dữ liệu và nhãn
    X_raw = data['clean_text']  # Văn bản đã tiền xử lý
    y = data['target']  # Nhãn cảm xúc (0 hoặc 1)

    # Chuyển đổi dữ liệu thô thành ma trận TF-IDF
    X_tfidf = tfidf_vectorizer.transform(X_raw)

    # Giảm số chiều với PCA
    X_tfidf_pca = pca.transform(X_tfidf.toarray())

    # Chia dữ liệu thành tập train/test
    X_train, X_test, y_train, y_test = train_test_split(X_tfidf_pca, y, test_size=0.2, random_state=42)

    # Dự đoán và đánh giá
    y_pred_svm = model_svm.predict(X_test)

    # Hiển thị báo cáo đánh giá
    print("Classification Report:")
    print(classification_report(y_test, y_pred_svm, target_names=["Negative", "Positive"]))

    # Vẽ confusion matrix
    conf_matrix = confusion_matrix(y_test, y_pred_svm, labels=[0, 1])
    disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=["Negative", "Positive"])
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.show()


def plot_classification_report(report):

    # Chuyển đổi thành DataFrame
    labels = list(report.keys())
    precision = [report[label]['precision'] for label in labels]
    recall = [report[label]['recall'] for label in labels]
    f1_score = [report[label]['f1-score'] for label in labels]

    df = pd.DataFrame({
        'precision': precision,
        'recall': recall,
        'f1-score': f1_score
    }, index=labels)

    # Vẽ biểu đồ
    fig, ax = plt.subplots(figsize=(8, 6))
    df.plot(kind='bar', ax=ax)

    # Thêm nhãn và tiêu đề
    ax.set_title('Classification Report Metrics')
    ax.set_ylabel('Score')
    ax.set_xlabel('Class')
    ax.set_ylim(0, 1)
    ax.set_xticklabels(labels, rotation=0)
    plt.tight_layout()

    # Hiển thị biểu đồ
    plt.show()

def visualize_data_in_2d(model_dir, path_vectorizer, path_pca):
    # Tải vectorizer và PCA
    
    tfidf_vectorizer = joblib.load(os.path.join(model_dir, path_vectorizer))
    pca = joblib.load(os.path.join(model_dir, path_pca))
    # Tải dữ liệu
    data = pd.read_csv("sentiment140/processed_training.1600000_2.csv")

    if 'clean_text' in data.columns:
        data['clean_text'] = data['clean_text'].fillna('')  # Thay thế NaN bằng chuỗi rỗng

    data['target'] = data['target'].map({0: 0, 4: 1})  # Chuyển 4 thành 1

    # Tách dữ liệu và nhãn
    X_raw = data['clean_text']  # Văn bản đã tiền xử lý
    y = data['target']  # Nhãn cảm xúc (0 hoặc 1)

    # Chuyển đổi dữ liệu thô thành ma trận TF-IDF
    X_tfidf = tfidf_vectorizer.transform(X_raw)

    # Giảm số chiều với PCA (giảm xuống 2 chiều để có thể vẽ đồ thị)
    X_tfidf_pca = pca.transform(X_tfidf.toarray())  # Dữ liệu đã giảm chiều

    # Kiểm tra kích thước dữ liệu sau khi giảm chiều
    print(f"Shape of PCA-reduced data: {X_tfidf_pca.shape}")

    # Vẽ biểu đồ 2D
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(X_tfidf_pca[:, 0], X_tfidf_pca[:, 1], c=y, cmap=plt.cm.RdYlBu, edgecolors='k', alpha=0.7)
    plt.title("2D Visualization of Data after PCA")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.colorbar(scatter)
    plt.show()
    
# SVM
report = {
    'Negative': {'precision': 0.71, 'recall': 0.66, 'f1-score': 0.69, 'support': 159494},
    'Positive': {'precision': 0.69, 'recall': 0.73, 'f1-score': 0.71, 'support': 160506}
    # ,'accuracy': 0.70,
    # 'macro avg': {'precision': 0.70, 'recall': 0.70, 'f1-score': 0.70, 'support': 320000},
    # 'weighted avg': {'precision': 0.70, 'recall': 0.70, 'f1-score': 0.70, 'support': 320000}
}
#  native_bayes
report = {
    'Negative': {'precision': 0.76, 'recall': 0.76, 'f1-score': 0.76, 'support': 159494},
    'Positive': {'precision': 0.76, 'recall': 0.76, 'f1-score': 0.76, 'support': 160506}
}

model_dir = 'model/native_bayes/'
path_model =  'naive_bayes_model.pkl'
path_name = 'tfidf_vectorizer.pkl'
# show_coufusion_matrix(model_dir, path_model, path_name)
#plot_classification_report(report)

# show_coufusion_matrix_svm('model/SVM/', 'svm_model.pkl', 'tfidf_vectorizer.pkl', 'pca.pkl')

visualize_data_in_2d('model/SVM/', 'tfidf_vectorizer.pkl', 'pca.pkl')