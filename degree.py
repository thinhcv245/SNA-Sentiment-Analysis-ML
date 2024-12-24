import matplotlib.pyplot as plt
import numpy as np
import networkx as nx

def plot_degree_distribution(graph, title="Degree Distribution"):
    """
    Trực quan hóa phân phối bậc (degree distribution) của đồ thị.
    - graph: Đồ thị mạng xã hội (NetworkX Graph).
    - title: Tiêu đề biểu đồ.
    """
    # Lấy bậc của tất cả các nút
    degrees = [deg for _, deg in graph.degree()]
    
    # Tính tần suất của mỗi bậc
    unique_degrees, counts = np.unique(degrees, return_counts=True)

    # Trực quan hóa phân phối bậc
    plt.figure(figsize=(8, 6))
    plt.scatter(unique_degrees, counts, color="blue", alpha=0.6, edgecolor="k", s=50)
    plt.xscale("log")  # Log-log scale trên trục x
    plt.yscale("log")  # Log-log scale trên trục y
    plt.title(title, fontsize=15)
    plt.xlabel("Degree (log scale)", fontsize=12)
    plt.ylabel("Frequency (log scale)", fontsize=12)
    plt.grid(alpha=0.3, which="both")
    plt.show()

# Sử dụng
if __name__ == "__main__":
    print("Loading graph...")
    graph = nx.read_graphml("graph.graphml")
    # Kiểm tra số nút và số cạnh
    print(f"Number of nodes: {graph.number_of_nodes()}")
    print(f"Number of edges: {graph.number_of_edges()}")

    print("Vẽ phân phối bậc của đồ thị...")
    plot_degree_distribution(graph, title="Degree Distribution of Social Network")
