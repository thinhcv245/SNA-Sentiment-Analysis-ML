from config import *
from services.preprocess import load_and_preprocess
from services.graph_builder import build_graph
from services.depic_model import depic_propagation
from services.evaluation import evaluate
import random
import networkx as nx
import numpy as np
import pandas as pd

if __name__ == "__main__":
    print("Loading graph...")
    graph = nx.read_graphml("g27k.graphml")

    # Kiểm tra số nút và số cạnh
    print(f"Number of nodes: {graph.number_of_nodes()}")
    print(f"Number of edges: {graph.number_of_edges()}")
    # Kiểm tra degree của các nút
    degree = [deg for _, deg in graph.degree()]
    print(f"Average degree: {np.mean(degree):.2f}")

    # Lưu lại cảm xúc ban đầu làm ground truth
    ground_truth = {node: graph.nodes[node].get("sentiment", 0) for node in graph.nodes}

    # Bước 3: Thực hiện lan truyền cảm xúc
    print("Đang thực hiện mô hình DepIC...")
    graph = depic_propagation(graph=graph, alpha=ALPHA, beta_pos=BETA_POS, beta_neg=BETA_NEG, time_decay=TIME_DECAY, max_steps=MAX_STEPS)
    #graph = propagate_emotion(graph, alpha=0.9, beta=0.5, lambda_=0.1, steps=10)

    # Bước 4: Đánh giá mô hình
    print("Đang đánh giá mô hình...")
    precision, recall, f1 = evaluate(graph, ground_truth)

    # Hiển thị kết quả
    print(f"Precision: {precision:.2f}, Recall: {recall:.2f}, F1-score: {f1:.2f}")

    # Kiểm tra trạng thái cảm xúc trước và sau lan truyền
    for node in list(graph.nodes)[:10]:  # Hiển thị 10 nút đầu tiên
        print(f"User: {node}, Before: {ground_truth[node]}, After: {graph.nodes[node]['sentiment']}")
