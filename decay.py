from config import *
import matplotlib.pyplot as plt
import networkx as nx
from services.graph_builder import build_graph
from services.depic_model import depic_propagation, fPOW, fNOR, fRAY
from services.preprocess import load_and_preprocess
import numpy as np
from scipy.interpolate import make_interp_spline

def plot_time_decay_influence(graphs_over_steps, decay_functions, title="Effect of Time Decay on Emotion", steps_to_plot=None):
    """
    Vẽ biểu đồ ảnh hưởng của các hàm suy giảm thời gian đến cảm xúc tích cực.
    - graphs_over_steps: Danh sách các trạng thái đồ thị qua từng bước, ứng với từng hàm suy giảm.
    - decay_functions: Danh sách các tên hàm suy giảm.
    - title: Tiêu đề biểu đồ.
    - steps_to_plot: Danh sách các bước cụ thể để đánh dấu điểm (đường vẫn mượt qua tất cả các bước).
    """
    plt.figure(figsize=(10, 6))
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
        marker = markers[i % len(markers)]
        plt.plot(
            steps_smooth,
            positive_ratios_smooth,
            label=f"{decay_functions[i]}",
            linestyle="-"
        )
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
    decay_functions = {
        "Power-Law": fPOW,
        "Normal": fNOR,
        "Rayleigh": fRAY
    }
    graphs_over_steps = []
    for decay_name, decay_function in decay_functions.items():
        print(f"Lan truyền với hàm suy giảm {decay_name}")
        graph = build_graph(PROCESSED_PATH)
        graphs_for_decay = []
        for step in range(MAX_STEPS):
            graph = depic_propagation(graph=graph, alpha=ALPHA, beta_pos=BETA_POS, beta_neg=BETA_NEG, decay_function=decay_function, max_steps=1)
            graphs_for_decay.append(graph.copy())
            print(f'{step + 1}/{MAX_STEPS}')
        graphs_over_steps.append(graphs_for_decay)

    print("Vẽ ảnh hưởng của hàm suy giảm thời gian...")
    plot_time_decay_influence(graphs_over_steps, list(decay_functions.keys()))
