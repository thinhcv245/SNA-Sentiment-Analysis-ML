import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import community as community_louvain  # Đảm bảo đang sử dụng thư viện đúng

# Tải đồ thị
print("Loading graph...")
G = nx.read_graphml("graph.graphml")

# Chuyển đồ thị thành không hướng
G = G.to_undirected()

# Bước 1: Phát hiện cộng đồng bằng thuật toán Louvain
def detect_communities(graph):
    # Tính toán cộng đồng sử dụng thuật toán Louvain
    partition = community_louvain.best_partition(graph)  # Sử dụng community_louvain
    return partition

# Bước 2: Mô phỏng lan truyền cảm xúc trong các cộng đồng
def propagate_emotion(graph, alpha, steps, partition):
    for node in graph.nodes:
        if 'emotion' not in graph.nodes[node]:
            graph.nodes[node]['emotion'] = 0
    positive_proportion = []

    for _ in range(steps):
        new_emotions = {}
        for node in graph.nodes:
            if graph.nodes[node]['emotion'] > 0:
                continue

            # Chỉ lan truyền trong cùng cộng đồng
            community_id = partition[node]
            positive_neighbors = sum(
                1 for neighbor in graph.neighbors(node)
                if graph.nodes[neighbor]['emotion'] > 0 and partition[neighbor] == community_id
            )
            
            influence = alpha * positive_neighbors
            if influence > np.random.rand():
                new_emotions[node] = 1

        # Cập nhật cảm xúc
        for node, emotion in new_emotions.items():
            graph.nodes[node]['emotion'] = emotion

        # Tính toán tỷ lệ tích cực
        positive_count = sum(1 for _, data in graph.nodes(data=True) if data['emotion'] == 1)
        positive_proportion.append(positive_count / graph.number_of_nodes())

    return positive_proportion

# Bước 3: Mô phỏng cho các giá trị alpha khác nhau
alphas = [0.1, 0.3, 0.5, 0.7, 1.0]
steps = 10
results = {}

# Phát hiện cộng đồng
partition = detect_communities(G)

for alpha in alphas:
    results[alpha] = propagate_emotion(G.copy(), alpha, steps, partition)

# Bước 4: Vẽ đồ thị kết quả
plt.figure(figsize=(10, 6))
for alpha, proportions in results.items():
    plt.plot(range(steps), proportions, label=f'Alpha = {alpha}')

plt.xlabel('Steps')
plt.ylabel('Proportion of Positive Emotions')
plt.title('Effect of Activation Intensity (Alpha) on Positive Emotion Spread')
plt.legend()
plt.grid()
plt.show()
