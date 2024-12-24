import numpy as np
import networkx as nx

def depic_propagation(graph, alpha, beta_pos, beta_neg, time_decay, max_steps):
    """
    Lan truyền cảm xúc sử dụng mô hình DepIC.
    """
    for step in range(max_steps):
        new_sentiments = {}

        # Lan truyền cảm xúc
        for node in graph.nodes:
            current_emotion = graph.nodes[node]["sentiment"]
            neighbors = list(graph.neighbors(node))

            for neighbor in neighbors:
                neighbor_emotion = graph.nodes[neighbor]["sentiment"]
                weight = graph[node][neighbor].get('weight', 1)

                # Tính ảnh hưởng giữa các nút (cả tích cực và tiêu cực)
                influence = alpha * (beta_pos if current_emotion > 0 else beta_neg)
                delta_emotion = influence * (current_emotion - neighbor_emotion)
                delta_emotion *= np.exp(-time_decay * step)  # Suy giảm theo thời gian

                # Tích lũy cảm xúc
                if neighbor not in new_sentiments:
                    new_sentiments[neighbor] = graph.nodes[neighbor]["sentiment"]
                new_sentiments[neighbor] += delta_emotion

        # Cập nhật cảm xúc
        changes = 0
        for node, sentiment in new_sentiments.items():
            new_sentiment = np.clip(sentiment, -1, 1)  # Giới hạn cảm xúc
            if graph.nodes[node]["sentiment"] != new_sentiment:
                changes += 1
            graph.nodes[node]["sentiment"] = new_sentiment

        # Theo dõi tiến trình lan truyền
        print(f"Step {step + 1}/{max_steps}: {changes} nodes changed sentiment.")

        # Dừng nếu không có thay đổi
        if changes == 0:
            print("No further changes, stopping propagation.")
            break

    return graph
