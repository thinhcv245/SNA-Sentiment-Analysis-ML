from sklearn.metrics import precision_score, recall_score, f1_score, classification_report

def evaluate(graph, ground_truth):
    """
    Đánh giá mô hình bằng Precision, Recall, F1-score.
    """
    # Dự đoán từ đồ thị sau lan truyền
    predictions = [graph.nodes[node]["sentiment"] for node in graph.nodes]

    # Chuyển ground truth thành danh sách theo cùng thứ tự với dự đoán
    truth = [ground_truth[node] for node in graph.nodes]

    # Chuyển đổi predictions thành nhãn rời rạc (-1, 1)
    predictions = [1 if p > 0 else -1 for p in predictions]

    # Kiểm tra nhãn trong dữ liệu
    unique_truth = set(truth)
    unique_predictions = set(predictions)
    print(f"Unique labels in truth: {unique_truth}")
    print(f"Unique labels in predictions: {unique_predictions}")

    # Tính toán các chỉ số
    precision = precision_score(truth, predictions, average="weighted", zero_division=1)
    recall = recall_score(truth, predictions, average="weighted", zero_division=1)
    f1 = f1_score(truth, predictions, average="weighted", zero_division=1)

    # Hiển thị báo cáo chi tiết
    report = classification_report(truth, predictions, target_names=[str(label) for label in unique_truth])
    print("\nClassification Report:\n", report)

    return precision, recall, f1
