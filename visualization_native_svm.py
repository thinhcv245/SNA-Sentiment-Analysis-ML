import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Load datasets (update paths as needed)
data_1 = pd.read_csv("sentiment140/elon_musk.csv")
data_2 = pd.read_csv("sentiment140/processed_training.1600000_2.csv")

# Function to clean and preprocess data
def preprocess_data(data):
    # Ensure columns 'clean_text' and 'target' exist
    if 'clean_text' not in data.columns or 'target' not in data.columns:
        raise ValueError("Dataset must contain 'clean_text' and 'target' columns")
    
    # Replace NaN values in 'clean_text' with an empty string
    data['clean_text'] = data['clean_text'].fillna("")
    
    # Drop rows where 'target' is NaN
    data = data.dropna(subset=['target'])
    
    # Convert 'target' column to integers if necessary
    if data['target'].dtype != np.int64 and data['target'].dtype != np.int32:
        data['target'] = data['target'].astype(int)
    
    return data

# Function to evaluate models
def evaluate_models(data, vectorizer, svd, model_naive, model_svm):
    # Preprocess the data
    data = preprocess_data(data)
    
    # Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        data['clean_text'], data['target'], test_size=0.2, random_state=42
    )

    # Vectorize text using TF-IDF
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    # Train and evaluate Naive Bayes (on original TF-IDF data)
    model_naive.fit(X_train_tfidf, y_train)
    y_pred_naive = model_naive.predict(X_test_tfidf)
    acc_naive = accuracy_score(y_test, y_pred_naive)

    # Reduce dimensionality with TruncatedSVD
    X_train_svd = svd.fit_transform(X_train_tfidf)
    X_test_svd = svd.transform(X_test_tfidf)

    # Train and evaluate SVM (on reduced SVD data)
    model_svm.fit(X_train_svd, y_train)
    y_pred_svm = model_svm.predict(X_test_svd)
    acc_svm = accuracy_score(y_test, y_pred_svm)

    return acc_naive, acc_svm

# Initialize vectorizer, SVD, models
vectorizer = TfidfVectorizer(max_features=10000)  # Limit to 10,000 features
svd = TruncatedSVD(n_components=100, random_state=42)  # Reduce to 100 components
model_naive = MultinomialNB()
model_svm = LinearSVC(C=1.0, max_iter=1000, dual=False)

# Evaluate models on each dataset
results = {}
dataset_names = ["Elon Musk Tweets", "Sentiment140"]  # Names for datasets

for i, (data, name) in enumerate(zip([data_1, data_2], dataset_names), start=1):
    try:
        acc_naive, acc_svm = evaluate_models(data, vectorizer, svd, model_naive, model_svm)
        results[name] = {
            "Naive Bayes": acc_naive,
            "SVM": acc_svm
        }
    except Exception as e:
        print(f"Error processing {name}: {e}")
        results[name] = {
            "Naive Bayes": None,
            "SVM": None
        }

# Prepare data for visualization
labels = list(results.keys())  # Dataset names
naive_scores = [results[label]["Naive Bayes"] if results[label]["Naive Bayes"] is not None else 0 for label in labels]
svm_scores = [results[label]["SVM"] if results[label]["SVM"] is not None else 0 for label in labels]

x = np.arange(len(labels))  # Label locations
width = 0.35  # Bar width

# Create bar plot
fig, ax = plt.subplots(figsize=(10, 6))

# Plot bars
bars1 = ax.bar(x - width/2, naive_scores, width, label='Naive Bayes',color='skyblue')
bars2 = ax.bar(x + width/2, svm_scores, width, label='SVM',color='#ff7e70')

# Add labels, title, and legend
ax.set_xlabel('Datasets')
ax.set_ylabel('Accuracy')
ax.set_title('Comparison of Naive Bayes and SVM with SVD on Different Datasets')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()

# Display plot
plt.tight_layout()
plt.show()