import numpy as np
import networkx as nx

def fPOW(t, t_u=0, c=1, theta=0.1):
    """Hàm suy giảm Power-Law."""
    if t <= t_u:
        return 0  # Không có suy giảm trước thời điểm kích hoạt
    return ((t - t_u) / c) ** (-theta + 1)

def fNOR(t, t_u=0, micro=1, sigma=1):
    """Hàm suy giảm Normal (Gaussian)."""
    if t <= t_u:
        return 0  # Không có suy giảm trước thời điểm kích hoạt
    return (1 / (np.sqrt(2 * np.pi) * sigma)) * np.exp(-((t - t_u - micro) ** 2) / (2 * sigma ** 2))

def fRAY(t, t_u=0, theta=0.1):
    """Hàm suy giảm Rayleigh."""
    if t <= t_u:
        return 0  # Không có suy giảm trước thời điểm kích hoạt
    return theta * (t - t_u) * np.exp(-0.5 * theta * (t - t_u) ** 2)


def depic_propagation(graph, alpha, beta_pos, beta_neg, time_decay=None, max_steps=10, decay_function=None):
    """
    Lan truyền cảm xúc sử dụng mô hình DepIC.
    - Nếu decay_function không được truyền, sử dụng hàm suy giảm mũ mặc định.
    """
    # Xác định hàm suy giảm mặc định (hàm mũ)
    def default_decay(t):
        return np.exp(-time_decay * t) if time_decay is not None else 1

    # Sử dụng hàm suy giảm được truyền vào hoặc mặc định
    decay_func = decay_function if decay_function is not None else default_decay

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
                delta_emotion *= decay_func(step)  # Sử dụng hàm suy giảm

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
