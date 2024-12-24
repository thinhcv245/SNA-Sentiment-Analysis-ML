import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
from config import *
from services.depic_model import depic_propagation

def plot_alpha_influence(graphs_over_steps, alphas, title="Effect of Alpha on Positive Emotion"):
    """
    Vẽ biểu đồ ảnh hưởng của tham số alpha đến tỷ lệ cảm xúc tích cực.
    - graphs_over_steps: Danh sách các trạng thái đồ thị qua từng bước, ứng với từng alpha.
    - alphas: Danh sách các giá trị alpha.
    - title: Tiêu đề biểu đồ.
    """
    plt.figure(figsize=(10, 6))

    for i, graphs in enumerate(graphs_over_steps):
        positive_ratios = []
        for graph in graphs:
            emotions = nx.get_node_attributes(graph, "sentiment")
            total_nodes = len(emotions)
            positive_nodes = sum(1 for emotion in emotions.values() if emotion > 0)
            positive_ratios.append(positive_nodes / total_nodes)

        plt.plot(range(len(graphs)), positive_ratios, label=f"Alpha={alphas[i]}", marker="o")

    plt.title(title, fontsize=15)
    plt.xlabel("Steps", fontsize=12)
    plt.ylabel("Proportion of Positive Emotion", fontsize=12)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()

# Ví dụ sử dụng
if __name__ == "__main__":
    alphas = [0.1, 0.3, 0.5, 0.7, 0.9]  # Các giá trị alpha để thử nghiệm
    graphs_over_steps = []

    for alpha in alphas:
        print(f"Lan truyền với alpha={alpha}")
        print("Loading graph...")
        graph = nx.read_graphml("graph.graphml")
        # Kiểm tra số nút và số cạnh
        print(f"Number of nodes: {graph.number_of_nodes()}")
        print(f"Number of edges: {graph.number_of_edges()}")
        graphs_for_alpha = []
        for step in range(MAX_STEPS):
            graph = depic_propagation(graph, alpha, BETA_POS, BETA_NEG, TIME_DECAY, 1)
            graphs_for_alpha.append(graph.copy())
        graphs_over_steps.append(graphs_for_alpha)

    print("Vẽ ảnh hưởng của alpha...")
    plot_alpha_influence(graphs_over_steps, alphas)
