import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
from config import *
from services.depic_model import depic_propagation

from scipy.interpolate import make_interp_spline

def plot_alpha_influence(graphs_over_steps, alphas, title="Effect of Alpha on Positive Emotion", steps_to_plot=None):
    """
    Vẽ biểu đồ ảnh hưởng của tham số alpha đến tỷ lệ cảm xúc tích cực với đường cong mượt mà.
    - graphs_over_steps: Danh sách các trạng thái đồ thị qua từng bước, ứng với từng alpha.
    - alphas: Danh sách các giá trị alpha.
    - title: Tiêu đề biểu đồ.
    - steps_to_plot: Danh sách các bước cụ thể để đánh dấu điểm (đường vẫn mượt qua tất cả các bước).
    """
    plt.figure(figsize=(10, 6))

    # Danh sách marker cho các line
    markers = ["o", "s", "D", "^", "v", "x", "+", "*", "p", "h"]

    for i, graphs in enumerate(graphs_over_steps):
        positive_ratios = []
        for graph in graphs:
            emotions = nx.get_node_attributes(graph, "sentiment")
            total_nodes = len(emotions)
            positive_nodes = sum(1 for emotion in emotions.values() if emotion > 0)
            positive_ratios.append(positive_nodes / total_nodes)

        # Tạo danh sách các bước
        steps = np.arange(len(graphs))

        # Làm mượt bằng B-spline
        steps_smooth = np.linspace(steps[0], steps[-1], 500)  # Tạo 500 điểm mượt
        spline = make_interp_spline(steps, positive_ratios, k=3)
        positive_ratios_smooth = spline(steps_smooth)

        # Chọn marker khác nhau cho mỗi alpha
        marker = markers[i % len(markers)]

        # Vẽ đường cong mượt với marker trên line
        plt.plot(
            steps_smooth,
            positive_ratios_smooth,
            label=f"α={alphas[i]}",
            linestyle="-"
        )

        # Đánh dấu điểm tại các steps_to_plot (nếu có)
        if steps_to_plot:
            selected_steps = [step for step in steps_to_plot if step < len(graphs)]
            selected_ratios = [positive_ratios[step] for step in selected_steps]
            plt.scatter(selected_steps, selected_ratios, marker=marker)

    plt.title(title, fontsize=15)
    plt.xlabel("Steps", fontsize=12)
    plt.ylabel("Proportion of Positive Emotion", fontsize=12)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()

# Ví dụ sử dụng
if __name__ == "__main__":
    alphas = [0.2, 0.5, 0.7, 0.9]  # Các giá trị alpha để thử nghiệm
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
            graph = depic_propagation(graph=graph, alpha=alpha, beta_pos=BETA_POS, beta_neg=BETA_NEG, time_decay=TIME_DECAY, max_steps=1)
            graphs_for_alpha.append(graph.copy())
            print(f'{step + 1}/{MAX_STEPS}')
        graphs_over_steps.append(graphs_for_alpha)

    print("Vẽ ảnh hưởng của alpha...")
    steps_to_plot = [step for step in range(1, MAX_STEPS + 1) if step % 10 == 0]
    plot_alpha_influence(graphs_over_steps=graphs_over_steps, alphas=alphas, steps_to_plot=steps_to_plot)
