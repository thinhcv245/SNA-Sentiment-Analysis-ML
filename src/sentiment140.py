import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import re
import EoN
import os

# === 1. Đọc và xử lý dữ liệu ===
# Load dataset
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data = pd.read_csv(os.path.join(base_dir, "data/raw/sentiment140.csv"), encoding='latin-1', header=None)
data.columns = ["target", "id", "date", "flag", "user", "text"]
data = data[["target", "user", "text"]]
data['target'] = data['target'].map({0: 'negative', 2: 'neutral', 4: 'positive'})

# === 2. Xây dựng danh sách các cạnh từ mentions ===
# Tách mentions trước khi làm sạch văn bản
edges = []
for index, row in data.iterrows():
    mentions = re.findall(r'@\w+', row['text'])
    for mention in mentions:
        edges.append((row['user'], mention.strip('@')))

# Làm sạch văn bản sau khi trích xuất mentions
def clean_text(text):
    text = re.sub(r'http\S+|www\S+', '', text)  # Loại bỏ URL
    text = re.sub(r'@\w+', '', text)  # Loại bỏ mentions
    text = re.sub(r'#\w+', '', text)  # Loại bỏ hashtags
    text = re.sub(r'\W', ' ', text)  # Loại bỏ ký tự đặc biệt
    text = text.lower().strip()
    return text

data['text'] = data['text'].apply(clean_text)
# edges_df = pd.DataFrame(edges)
# print(f"Số lượng edges: {len(edges)}")
# edges_df.to_csv(os.path.join(base_dir, "data/processed/sentiment140/edges.csv"), index=False)

# === 3. Xây dựng mạng xã hội ===
# Tạo đồ thị từ danh sách cạnh
G = nx.DiGraph()
G.add_edges_from(edges)
print("Số lượng edge:", len(edges))
# Thêm thuộc tính cảm xúc cho từng node
emotion_map = data.set_index('user')['target'].to_dict()
nx.set_node_attributes(G, emotion_map, 'emotion')

# === 4. Thống kê và phân tích mạng ===
print("Số lượng node:", G.number_of_nodes())
print("Số lượng cạnh:", G.number_of_edges())
print("Hệ số clustering trung bình:", nx.average_clustering(G.to_undirected()))

# === 5. Mô phỏng lan truyền cảm xúc ===
# Lựa chọn các node ban đầu (seed nodes)
seed_nodes = [node for node, attr in G.nodes(data=True) if attr.get('emotion') == 'positive']

# Mô phỏng lan truyền cảm xúc bằng mô hình Independent Cascade
t, S, I, R = EoN.fast_SIR(G, tau=0.1, gamma=0.01, initial_infecteds=seed_nodes)

# === 6. Trực quan hóa ===
# Vẽ biểu đồ số lượng người lan truyền cảm xúc theo thời gian
plt.figure(figsize=(10, 6))
plt.plot(t, S, label='Susceptible')
plt.plot(t, I, label='Infected (Emotion Spreading)')
plt.plot(t, R, label='Recovered')
plt.xlabel('Time')
plt.ylabel('Number of Users')
plt.title('Emotion Spread over Time')
plt.legend()
plt.show()

# Vẽ mạng xã hội với cảm xúc
color_map = {'positive': 'green', 'negative': 'red', 'neutral': 'blue'}
node_colors = [color_map.get(G.nodes[node].get('emotion', 'neutral'), 'grey') for node in G]

plt.figure(figsize=(12, 8))
pos = nx.spring_layout(G, k=0.15, seed=42)  # Bố cục đồ thị
nx.draw(
    G,
    pos,
    node_color=node_colors,
    with_labels=False,
    node_size=50,
    edge_color="gray",
    alpha=0.7
)
plt.title("Emotion Network Visualization")
plt.show()