import pandas as pd
import os
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from collections import Counter
from itertools import combinations
from datetime import datetime

# Tạo thư mục visualization/gephi nếu chưa tồn tại

def save_visualization(fig, output_dir, filename):
    # Tạo thư mục nếu chưa tồn tại
    os.makedirs(output_dir, exist_ok=True)
    
    # Tạo đường dẫn đầy đủ cho tệp hình ảnh
    file_path = os.path.join(output_dir, filename)
    
    # Lưu hình ảnh vào thư mục
    fig.savefig(file_path)
    print(f"Image saved at: {file_path}")
def visualization_gephi():
    output_dir = "visualization/gephi"
    os.makedirs(output_dir, exist_ok=True)
    # Load data
    path_output = "sentiment140/processed_training.1600000_2.csv"

    data = pd.read_csv(path_output)

    # Gắn nhãn cảm xúc
    sentiment_map = {0: "Negative", 4: "Positive"}

    # Tạo tệp nodes.csv
    nodes = pd.DataFrame({
        'Id': [1, 2],
        'Label': ['Positive', 'Negative'],
        'Sentiment': ['Positive', 'Negative']
    })
    nodes.to_csv('visualization/gephi/nodes.csv', index=False)

    # Tạo tệp edges.csv
    edges = []
    for _, row in data.iterrows():
        target = 1 if row['target'] == 4 else 2
        edges.append({
            'Source': row['user'],
            'Target': target,
            'Type': 'Directed',
            'Weight': 1
        })
    edges_df = pd.DataFrame(edges)
    edges_df.to_csv('visualization/gephi/edges.csv', index=False)
def visualization_gephi_v2(data):
    output_dir = "visualization/gephi2"
    os.makedirs(output_dir, exist_ok=True)
    # Xử lý từng câu thành danh sách từ
    data['words'] = data['clean_text'].apply(lambda x: str(x).split() if isinstance(x, str) else [])

    # Đếm đồng xuất hiện
    co_occurrences = Counter()
    for words in data['words']:
        for pair in combinations(words, 2):
            co_occurrences[pair] += 1

    # Chuẩn bị dữ liệu cho Gephi
    nodes = set()
    edges = []
    for (word1, word2), weight in co_occurrences.items():
        nodes.add(word1)
        nodes.add(word2)
        edges.append({'Source': word1, 'Target': word2, 'Weight': weight})

    # Lưu nodes.csv
    nodes_df = pd.DataFrame({'Id': list(nodes), 'Label': list(nodes)})
    nodes_df.to_csv('visualization/gephi2/nodes.csv', index=False)

    # Lưu edges.csv
    edges_df = pd.DataFrame(edges)
    edges_df.to_csv('visualization/gephi2/edges.csv', index=False)
    print("Files saved for Gephi!")


def visualization_bart_chat(data):
    # Gắn nhãn cảm xúc
    sentiment_map = {0: "Negative", 4: "Positive"}
    data['Sentiment'] = data['target'].map(sentiment_map)

    # Đếm số lượng cảm xúc
    sentiment_counts = data['Sentiment'].value_counts()

    # Vẽ biểu đồ cột
    fig, ax = plt.subplots(figsize=(8, 6))
    sentiment_counts.plot(kind='bar', color=['#f86145', '#73c181'], ax=ax)
    ax.set_title('Sentiment Distribution')
    ax.set_xlabel('Sentiment')
    ax.set_ylabel('Frequency')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
    
    # Lưu biểu đồ
    save_visualization(fig, "visualization/images", "sentiment_distribution.png")
    plt.close(fig)  # Đóng figure để giải phóng bộ nhớ
def visualization_pie_chart(data):
    # Gắn nhãn cảm xúc
    sentiment_map = {0: "Negative", 4: "Positive"}
    data['Sentiment'] = data['target'].map(sentiment_map)

    # Đếm số lượng cảm xúc
    sentiment_counts = data['Sentiment'].value_counts()

    # Sử dụng plt.subplots() thay vì plt.figure()
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Vẽ biểu đồ hình tròn (Pie chart)
    sentiment_counts.plot(kind='pie', autopct='%1.1f%%', colors=['#f86145', '#73c181'], startangle=140, ax=ax)
    
    ax.set_title('Sentiment Proportion')
    ax.set_ylabel('')  # Loại bỏ nhãn trục Y
    
    # Lưu biểu đồ
    save_visualization(fig, "visualization/images", "sentiment_proportion_pie_chart.png")
    plt.close(fig)  # Đóng figure để giải phóng bộ nhớ
    
def visualization_wordcloud(data):
    sentiment_map = {0: "Negative", 4: "Positive"}
    data['Sentiment'] = data['target'].map(sentiment_map)
    data['clean_text'] = data['clean_text'].fillna("").astype(str)

    # Đếm số lượng cảm xúc
    positive_text = " ".join(data[data['Sentiment'] == "Positive"]['clean_text'])
    negative_text = " ".join(data[data['Sentiment'] == "Negative"]['clean_text'])

    # Tạo WordCloud cho Positive
    positive_wc = WordCloud(width=900, height=450, background_color='white', colormap='Greens').generate(positive_text)

    # Tạo WordCloud cho Negative
    negative_wc = WordCloud(width=900, height=450, background_color='white', colormap='Reds').generate(negative_text)

    # Vẽ WordCloud
   # Vẽ WordCloud
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    ax1.imshow(positive_wc, interpolation='bilinear')
    ax1.set_title('Positive Word Cloud', fontsize=18)
    ax1.axis('off')
    
    ax2.imshow(negative_wc, interpolation='bilinear')
    ax2.set_title('Negative Word Cloud', fontsize=18)
    ax2.axis('off')

    # Lưu WordCloud
    save_visualization(fig, "visualization/images", "wordcloud_v1.png")
    plt.close(fig)  # Đóng figure để giải phóng bộ nhớ
    
def visualization_sentence_length(data):
    # Tính độ dài câu
    
    data['sentence_length'] = data['clean_text'].apply(lambda x: len(str(x).split()) if isinstance(x, str) else 0)

    # Vẽ phân phối độ dài câu
    fig, ax = plt.subplots(figsize=(8, 6))

    data['sentence_length'].hist(bins=50, color='#4a918d', alpha=0.7)
    ax.set_title('Sentence Length Distribution')
    ax.set_xlabel('Number of Words')
    ax.set_ylabel('Frequency')
    save_visualization(fig, "visualization/images", "sentence_length.png")

    plt.close(fig)



def visualization_plot_average_sentence_length_by_hour(data):
    # Convert the 'date' column to datetime
    data['date'] = pd.to_datetime(data['date'], format='%a %b %d %H:%M:%S PDT %Y')

    # Extract the hour from the date column
    data['hour'] = data['date'].dt.hour

    # Calculate sentence length (number of words) in clean_text
    data['sentence_length'] = data['clean_text'].apply(lambda x: len(str(x).split()) if isinstance(x, str) else 0)

    # Group by hour and calculate the average sentence length
    avg_sentence_length_by_hour = data.groupby('hour')['sentence_length'].mean()

    # Plotting the average sentence length by hour
    fig, ax = plt.subplots(figsize=(10, 6))
    avg_sentence_length_by_hour.plot(kind='line', marker='o', color='b', ax=ax)
    
    # Adding title and labels
    ax.set_title('Average Sentence Length of Tweets by Hour of the Day', fontsize=14)
    ax.set_xlabel('Hour of the Day', fontsize=12)
    ax.set_ylabel('Average Sentence Length (Number of Words)', fontsize=12)
    
    # Formatting x-axis to show hours from 0 to 23
    ax.set_xticks(range(24))  # Display hours from 0 to 23
    ax.set_xticklabels([f'{i}:00' for i in range(24)], rotation=45)

    # Adding grid
    ax.grid(True)

    # Adding date and time information to the plot title
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")  # Get current date and time
    ax.text(0.5, 1.05, f"Generated on: {current_time}", ha='center', va='bottom', fontsize=10, color='grey')

    # Saving the plot
    save_visualization(fig, "visualization/images", "plot_average_sentence_length_by_hour.png")

    plt.close(fig)
    print(data[['clean_text', 'sentence_length']].head())


def visualization_plot_average_sentence_length_by_hour_circle(data):
    # Convert the 'date' column to datetime
    data['date'] = pd.to_datetime(data['date'], format='%a %b %d %H:%M:%S PDT %Y')

    # Extract the hour from the date column
    data['hour'] = data['date'].dt.hour

    # Calculate sentence length (number of words) in clean_text
    data['sentence_length'] = data['clean_text'].apply(lambda x: len(str(x).split()) if isinstance(x, str) else 0)

    # Group by hour and calculate the average sentence length
    avg_sentence_length_by_hour = data.groupby('hour')['sentence_length'].mean()

    # Plotting the average sentence length in a circular plot
    fig, ax = plt.subplots(figsize=(10, 6), subplot_kw={'projection': 'polar'})  # Polar plot
    hours = avg_sentence_length_by_hour.index
    avg_lengths = avg_sentence_length_by_hour.values

    # Create a circular plot
    ax.plot(hours * (2 * 3.14159 / 24), avg_lengths, marker='o', color='b')  # Convert hours to radians for circular plot
    ax.fill(hours * (2 * 3.14159 / 24), avg_lengths, color='b', alpha=0.3)  # Fill the area for better visualization
    
    # Set labels and title
    ax.set_title('Average Sentence Length of Tweets by Hour of the Day', fontsize=14)
    ax.set_xlabel('Hour of the Day', fontsize=12)
    ax.set_ylabel('Average Sentence Length (Number of Words)', fontsize=12)

    # Formatting x-axis labels (the hour labels)
    ax.set_xticks([i * (2 * 3.14159 / 24) for i in range(24)])
    ax.set_xticklabels([f'{i}:00' for i in range(24)], fontsize=10)

    # Adding grid
    ax.grid(True)

    # Adding date and time information to the plot title
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")  # Get current date and time
    ax.text(0.5, 1.05, f"Generated on: {current_time}", ha='center', va='bottom', fontsize=10, color='grey')

    # Saving the plot
    save_visualization(fig, "visualization/images", "plot_average_sentence_length_by_hour_polar.png")

    plt.close(fig)

    
path_output = "sentiment140/processed_training.1600000.csv"
data = pd.read_csv(path_output)
# visualization_bart_chat(data)
# visualization_pie_chart(data)
visualization_wordcloud(data)
# visualization_sentence_length(data)
# visualization_gephi_v2(data)
# visualization_plot_average_sentence_length_by_hour(data)
# visualization_plot_average_sentence_length_by_hour_circle(data)
