import pandas as pd
from graph_tool.all import *

# === 4. Xác minh bằng cách tải lại và kiểm tra ===
g = load_graph('social_network.gt')
print("Số lượng node:", g.num_vertices())
print("Số lượng cạnh:", g.num_edges())

# Hệ số clustering trung bình
clustering = global_clustering(g)
print("Hệ số clustering trung bình:", clustering)

# === 5. Vẽ mạng xã hội ===
def draw_subgraph(g, title, output_path):
    # Định nghĩa bản đồ màu cho các cảm xúc
    color_map = {'positive': [0.0, 1.0, 0.0, 1.0], 'negative': [1.0, 0.0, 0.0, 1.0], 'neutral': [0.0, 0.0, 1.0, 1.0]}
    
    # Tạo thuộc tính màu sắc cho từng đỉnh
    v_color = g.new_vertex_property("vector<float>")
    for v in g.vertices():
        emotion = g.vertex_properties["emotion"][v]
        v_color[v] = color_map.get(emotion, [0.5, 0.5, 0.5, 1.0])  # Mặc định là màu xám

    # Bố cục đồ thị
    pos = sfdp_layout(g)
    
    # Vẽ và lưu đồ thị
    graph_draw(
        g,
        pos=pos,
        vertex_fill_color=v_color,
        vertex_size=30,
        edge_color="gray",
        output_size=(2000, 2000),
        output=output_path,
         bg_color=(1.0, 1.0, 1.0, 1.0)
    )
    print(f"Đồ thị đã được lưu tại: {output_path}")

# === 6. Giảm thiểu mạng theo độ kết nối (Degree-based Filtering) ===
degree_threshold = 10  # Ngưỡng độ kết nối
important_nodes = [v for v in g.vertices() if v.out_degree() >= degree_threshold]

# Tạo subgraph với những node quan trọng
important_subgraph = GraphView(g, vfilt=lambda v: v in important_nodes)
print("Số lượng node trong subgraph quan trọng:", important_subgraph.num_vertices())

# Vẽ subgraph
draw_subgraph(important_subgraph, "Important Node Subgraph (Degree Filtering)", "important_subgraph.png")
