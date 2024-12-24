import networkx as nx
import random
import matplotlib.pyplot as plt

# Tạo đồ thị mạng xã hội giả
G = nx.erdos_renyi_graph(100, 0.1)  # Mạng xã hội với 100 người dùng, xác suất kết nối là 0.1

# Khởi tạo cảm xúc cho mỗi người dùng: 0 = negative, 1 = positive
initial_emotions = {node: random.choice([0, 1]) for node in G.nodes()}

# Mô phỏng lan truyền cảm xúc
def propagate_emotions(G, emotions, steps=5):
    emotion_history = []
    for step in range(steps):
        new_emotions = emotions.copy()
        for node in G.nodes():
            neighbors = list(G.neighbors(node))
            if len(neighbors) == 0:
                continue
            neighbor_emotions = [emotions[neighbor] for neighbor in neighbors]
            # Nếu đa số bạn bè có cảm xúc tích cực, người này sẽ cảm thấy tích cực
            new_emotions[node] = 1 if sum(neighbor_emotions) > len(neighbors) / 2 else 0
        emotions = new_emotions
        emotion_history.append(emotions.copy())
    return emotion_history

# Chạy mô phỏng lan truyền cảm xúc
emotion_history = propagate_emotions(G, initial_emotions, steps=10)

# Trực quan hóa cảm xúc qua các bước
fig, axes = plt.subplots(2, 5, figsize=(15, 6))
for i, ax in enumerate(axes.flat):
    if i < len(emotion_history):
        emotions = emotion_history[i]
        color_map = ['red' if emotions[node] == 0 else 'green' for node in G.nodes()]
        nx.draw(G, node_color=color_map, ax=ax, with_labels=False, node_size=50, edge_color='gray', width=0.3)
        ax.set_title(f'Step {i+1}')
plt.tight_layout()
plt.show()
