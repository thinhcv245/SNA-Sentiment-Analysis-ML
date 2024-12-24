from config import *
from services.preprocess import load_and_preprocess
from services.graph_builder import build_graph
from services.depic_model import depic_propagation
from services.evaluation import evaluate
import random
import networkx as nx
import numpy as np
import pandas as pd

def propagate_emotion(G, alpha=0.9, beta=0.5, lambda_=0.1, steps=10):
    for _ in range(steps):
        updated_emotions = {}
        for node in G.nodes():
            current_emotion = G.nodes[node].get('emotion', 0)
            neighbors = list(G.neighbors(node))
            for neighbor in neighbors:
                weight = G[node][neighbor].get('weight', 1)
                prob = beta * (1 - (1 / (1 + lambda_ * weight)))
                if random.random() < prob:
                    if neighbor not in updated_emotions:
                        updated_emotions[neighbor] = 0
                    updated_emotions[neighbor] += alpha * current_emotion
        for node, new_emotion in updated_emotions.items():
            G.nodes[node]['emotion'] = new_emotion
    return G

if __name__ == "__main__":
    # Bước 1: Tải và xử lý dữ liệu
    print("Đang xử lý dữ liệu...")
    original_data = load_and_preprocess(DATA_PATH, PROCESSED_PATH)
    original_data.to_csv('original.csv', index=False)

    # Bước 2: Xây dựng đồ thị mạng xã hội
    print("Đang xây dựng mạng xã hội...")
    graph = build_graph(PROCESSED_PATH)
    nx.write_graphml(graph, "graph.graphml")

    # print("Loading graph...")
    # graph = nx.read_graphml("graph.graphml")

    # Kiểm tra số nút và số cạnh
    print(f"Number of nodes: {graph.number_of_nodes()}")
    print(f"Number of edges: {graph.number_of_edges()}")
    # Kiểm tra degree của các nút
    degree = [deg for _, deg in graph.degree()]
    print(f"Average degree: {np.mean(degree):.2f}")

    # Lưu lại cảm xúc ban đầu làm ground truth
    ground_truth = {row["user"]: row["sentiment"] for _, row in original_data.iterrows()}

    # Bước 3: Thực hiện lan truyền cảm xúc
    print("Đang thực hiện mô hình DepIC...")
    graph = depic_propagation(graph, ALPHA, BETA_POS, BETA_NEG, TIME_DECAY, MAX_STEPS)
    #graph = propagate_emotion(graph, alpha=0.9, beta=0.5, lambda_=0.1, steps=10)

    # Bước 4: Đánh giá mô hình
    print("Đang đánh giá mô hình...")
    precision, recall, f1 = evaluate(graph, ground_truth)

    # Hiển thị kết quả
    print(f"Precision: {precision:.2f}, Recall: {recall:.2f}, F1-score: {f1:.2f}")

    # Kiểm tra trạng thái cảm xúc trước và sau lan truyền
    for node in list(graph.nodes)[:10]:  # Hiển thị 10 nút đầu tiên
        print(f"User: {node}, Before: {ground_truth[node]}, After: {graph.nodes[node]['sentiment']}")
