import pandas as pd
import re
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from torch.optim import Adam

# Kiểm tra xem có GPU không
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hàm token hóa dữ liệu cho BERT
def tokenize_data(data, tokenizer):
    """
    Token hóa văn bản để đưa vào mô hình BERT.
    """
    inputs = tokenizer(list(data['clean_text']), padding=True, truncation=True, max_length=128, return_tensors="pt")
    return inputs

# Hàm chuẩn bị bộ dữ liệu TensorDataset
def create_dataset(inputs, labels):
    """
    Tạo TensorDataset từ dữ liệu đầu vào (input_ids, attention_mask) và nhãn cảm xúc.
    """
    dataset = TensorDataset(inputs['input_ids'], inputs['attention_mask'], torch.tensor(labels))
    return dataset

# Hàm huấn luyện mô hình BERT
def train_model(model, dataloader, optimizer, epochs=3):
    """
    Huấn luyện mô hình BERT.
    """
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch in dataloader:
            b_input_ids, b_attention_mask, b_labels = [item.to(device) for item in batch]

            optimizer.zero_grad()

            # Tiến hành forward pass
            outputs = model(b_input_ids, attention_mask=b_attention_mask, labels=b_labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1} - Loss: {total_loss / len(dataloader)}")

# Hàm dự đoán cảm xúc cho dữ liệu
def predict_sentiment(model, dataloader):
    """
    Dự đoán cảm xúc cho các tweet.
    """
    model.eval()
    predictions = []

    with torch.no_grad():
        for batch in dataloader:
            b_input_ids, b_attention_mask, _ = [item.to(device) for item in batch]
            outputs = model(b_input_ids, attention_mask=b_attention_mask)
            logits = outputs.logits
            predictions.extend(torch.argmax(logits, dim=-1).cpu().numpy())
    
    return predictions

# Hàm chính để thực hiện phân tích cảm xúc
def main():
    # Đường dẫn tới file dữ liệu
    path_output = "sentiment140/elon_musk_tweets_dataset.csv"

    # Tải tokenizer BERT
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # Giả sử load_data là một hàm tùy chỉnh để tải dữ liệu (ví dụ từ CSV)
    data = load_data(path_output)  # Cần xác định hàm này

    # Chia dữ liệu thành train và test set
    train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

    # Token hóa dữ liệu
    train_inputs = tokenize_data(train_data, tokenizer)
    test_inputs = tokenize_data(test_data, tokenizer)

    # Tạo TensorDataset cho train và test
    train_dataset = create_dataset(train_inputs, train_data['sentiment'].values)
    test_dataset = create_dataset(test_inputs, test_data['sentiment'].values)

    # Tạo DataLoader cho train và test
    train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    # Tải mô hình BERT đã được huấn luyện trước
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
    model.to(device)  # Chuyển mô hình sang GPU nếu có

    # Tạo optimizer
    optimizer = Adam(model.parameters(), lr=2e-5)

    # Huấn luyện mô hình
    train_model(model, train_dataloader, optimizer, epochs=3)

    # Dự đoán cảm xúc cho dữ liệu test
    predictions = predict_sentiment(model, test_dataloader)

    # Gắn nhãn cảm xúc cho test set
    test_data['predicted_sentiment'] = predictions

    # In kết quả
    print(test_data[['text', 'sentiment', 'predicted_sentiment']].head())

if __name__ == '__main__':
    main()
