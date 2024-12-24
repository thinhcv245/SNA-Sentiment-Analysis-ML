import networkx as nx
import pandas as pd

def build_graph(processed_path):
    """
    Xây dựng đồ thị mạng xã hội từ dữ liệu đã xử lý.
    """
    df = pd.read_csv(processed_path)
    
    # Tạo đồ thị
    graph = nx.DiGraph()

    # Thêm nút và cảm xúc
    for _, row in df.iterrows():
        graph.add_node(row["user"], sentiment=row["sentiment"])

    # Thêm cạnh (giả sử các người dùng retweet hoặc đề cập lẫn nhau)
    for _, row in df.iterrows():
        mentions = [word[1:] for word in row["text"].split() if word.startswith("@")]
        for mention in mentions:
            if mention in graph:
                graph.add_edge(row["user"], mention)

    return graph
