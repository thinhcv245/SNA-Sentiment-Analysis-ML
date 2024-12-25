import matplotlib.pyplot as plt
import random
from services.depic_model import depic_propagation
import networkx as nx
import community as community_louvain
from config import *

def visualize_graph(graph, title="Graph Visualization"):
    """
    Hàm trực quan hóa đồ thị với màu sắc dựa trên sentiment.
    """
    colors = []
    for node in graph.nodes(data=True):
        sentiment = node[1].get("sentiment", 0)
        if sentiment < 0:
            colors.append("red")
        elif sentiment > 0:
            colors.append("green")
        else:
            colors.append("blue")

    # Kích thước nút dựa trên out-degree
    sizes = [10 * (graph.out_degree(node) + 1) for node in graph.nodes]

    # Vẽ đồ thị
    pos = nx.spring_layout(graph, seed=42, k=0.05, iterations=100)  # Sử dụng spring layout
    
    # Vẽ các cạnh với đường cong
    nx.draw_networkx_edges(graph, pos, alpha=0.3, edge_color="gray", width=1, style="solid", connectionstyle="arc3,rad=0.2")
    
    # Vẽ các nút và nhãn
    nx.draw_networkx_nodes(graph, pos, node_color=colors, node_size=sizes, alpha=0.7)

    plt.title(title, fontsize=16)
    plt.axis('off')
    plt.show()

def largest_community_subgraph(graph):
    """
    Lọc thành phần liên thông lớn nhất và chọn cộng đồng lớn nhất.
    """
    # Lấy thành phần liên thông lớn nhất
    largest_cc = max(nx.weakly_connected_components(graph), key=len)
    subgraph = graph.subgraph(largest_cc).copy()

    # Chuyển thành đồ thị không có hướng
    undirected_subgraph = subgraph.to_undirected()

    # Phân cụm cộng đồng
    partition = community_louvain.best_partition(undirected_subgraph)

    # Tìm cộng đồng lớn nhất
    community_counts = {}
    for node, comm in partition.items():
        community_counts[comm] = community_counts.get(comm, 0) + 1
    largest_community = max(community_counts, key=community_counts.get)
    largest_community_nodes = [node for node, comm in partition.items() if comm == largest_community]

    community_subgraph = subgraph.subgraph(largest_community_nodes).copy()
    print(f"Community Subgraph: {community_subgraph.number_of_nodes()} nodes, {community_subgraph.number_of_edges()} edges")
    return community_subgraph

def connected_subgraph(graph, target_nodes=500):
    """
    Lấy mẫu khoảng target_nodes từ đồ thị, đảm bảo các node được kết nối với nhau.
    """
    # Bắt đầu từ node có degree lớn nhất (hoặc bất kỳ node nào)
    start_node = max(graph.degree, key=lambda x: x[1])[0]

    # BFS để thu thập các node gần nhất
    visited = set()
    queue = [start_node]

    while queue and len(visited) < target_nodes:
        current = queue.pop(0)
        if current not in visited:
            visited.add(current)
            # Thêm các láng giềng vào hàng đợi nếu chưa vượt quá target_nodes
            queue.extend(n for n in graph.neighbors(current) if n not in visited)

    # Tạo đồ thị con từ các node đã chọn
    sampled_subgraph = graph.subgraph(visited).copy()
    print(f"Sampled Subgraph: {sampled_subgraph.number_of_nodes()} nodes, {sampled_subgraph.number_of_edges()} edges")
    return sampled_subgraph

if __name__ == "__main__":
    print("Loading graph...")
    graph = nx.read_graphml("graph.graphml")

    # Lấy cộng đồng lớn nhất
    community_subgraph = largest_community_subgraph(graph)
    community_subgraph = connected_subgraph(community_subgraph, 300)
    print(f"Connected Subgraph: {community_subgraph.number_of_nodes()} nodes, {community_subgraph.number_of_edges()} edges")

    # Trực quan hóa ban đầu
    visualize_graph(community_subgraph, title="Initial Sentiment Distribution")

    # Lan truyền cảm xúc với DepIC
    community_subgraph = depic_propagation(graph=community_subgraph, alpha=ALPHA, beta_pos=BETA_POS, beta_neg=BETA_NEG, time_decay=TIME_DECAY, max_steps=10)
    # Trực quan hóa sau khi lan truyền
    visualize_graph(community_subgraph, title="Sentiment Distribution After DepIC - Step 10")
    community_subgraph = depic_propagation(graph=community_subgraph, alpha=ALPHA, beta_pos=BETA_POS, beta_neg=BETA_NEG, time_decay=TIME_DECAY, max_steps=10)
    visualize_graph(community_subgraph, title="Sentiment Distribution After DepIC - Step 20")