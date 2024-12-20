import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report

# Load datasets
data_1 = pd.read_csv("sentiment140/elon_musk.csv")
data_2 = pd.read_csv("sentiment140/processed_training.1600000_2.csv")

# Preprocessing function
def preprocess_data(data):
    if 'clean_text' not in data.columns or 'target' not in data.columns:
        raise ValueError("Dataset must contain 'clean_text' and 'target' columns")
    data['clean_text'] = data['clean_text'].fillna("")
    data = data.dropna(subset=['target'])
    if data['target'].dtype != np.int64 and data['target'].dtype != np.int32:
        data['target'] = data['target'].astype(int)
    return data

# Function to train models and generate detailed metrics
def evaluate_with_report(data, vectorizer, model_naive, model_svm):
    # Preprocess the data
    data = preprocess_data(data)
    X_train, X_test, y_train, y_test = train_test_split(
        data['clean_text'], data['target'], test_size=0.2, random_state=42
    )
    # Vectorize text
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    # Naive Bayes
    model_naive.fit(X_train_tfidf, y_train)
    y_pred_naive = model_naive.predict(X_test_tfidf)
    report_naive = classification_report(y_test, y_pred_naive, output_dict=True)

    # SVM
    model_svm.fit(X_train_tfidf, y_train)
    y_pred_svm = model_svm.predict(X_test_tfidf)
    report_svm = classification_report(y_test, y_pred_svm, output_dict=True)

    return report_naive, report_svm

# Initialize vectorizer and models
vectorizer = TfidfVectorizer(max_features=10000)
model_naive = MultinomialNB()
model_svm = LinearSVC(C=1.0, max_iter=1000, dual=False)

# Evaluate models on each dataset
datasets = {"Elon Musk": data_1, "Sentiment140": data_2}
all_reports = {}

for dataset_name, dataset in datasets.items():
    try:
        naive_report, svm_report = evaluate_with_report(dataset, vectorizer, model_naive, model_svm)
        all_reports[dataset_name] = {
            "Naive Bayes": naive_report,
            "SVM": svm_report
        }
    except Exception as e:
        print(f"Error processing {dataset_name}: {e}")
        all_reports[dataset_name] = {"Naive Bayes": None, "SVM": None}

# Convert report data into a structured table
# Convert report data into a structured table with improved format
rows = []
for dataset_name, reports in all_reports.items():
    for model_name, report in reports.items():
        if report:
            row = {
                "Dataset": dataset_name,
                "Model": model_name,
                "Precision": report['weighted avg']['precision'],
                "Recall": report['weighted avg']['recall'],
                "F1-Score": report['weighted avg']['f1-score'],
                "Support": int(report['weighted avg']['support']),
            }
            rows.append(row)

# Convert to DataFrame
df = pd.DataFrame(rows)

# Xuất kết quả ra file Excel
output_file = "visualization/model_comparison_results_improved.xlsx"
df.to_excel(output_file, index=False)

print(f"Kết quả đã được lưu vào file {output_file}")

