import pandas as pd
from graph_tool.all import *

# === 1. Đọc dữ liệu từ file CSV ===
# File nodes.csv: id, emotion (positive, negative, neutral)
# File edges.csv: source, target
nodes_df = pd.read_csv("data/raw/nodes.csv")
edges_df = pd.read_csv("data/raw/edges.csv")

# === 2. Xây dựng đồ thị với Graph-tool ===
# Tạo đồ thị
g = Graph(directed=True)

# Thêm thuộc tính emotion cho các node
v_emotion = g.new_vertex_property("string")
v_label = g.new_vertex_property("string")

# Tạo ánh xạ node ID với vertex trong Graph-tool
node_map = {}  # Ánh xạ từ id sang vertex
for _, row in nodes_df.iterrows():
    vertex = g.add_vertex()
    node_map[row['id']] = vertex
    v_label[vertex] = row['id']
    v_emotion[vertex] = row['emotion']

# Danh sách lưu các cạnh không hợp lệ
invalid_edges = []

# Thêm các cạnh
for _, row in edges_df.iterrows():
    source_id = row['source']
    target_id = row['target']
    if source_id in node_map and target_id in node_map:
        source = node_map[source_id]
        target = node_map[target_id]
        g.add_edge(source, target)
    else:
        invalid_edges.append((source_id, target_id))

# Gắn thuộc tính vào đồ thị
g.vertex_properties["emotion"] = v_emotion
g.vertex_properties["label"] = v_label

# Báo cáo các cạnh không hợp lệ
if invalid_edges:
    print("Các cạnh không hợp lệ (ID bị thiếu trong nodes.csv):")
    for edge in invalid_edges:
        print(edge)

# === 3. Lưu đồ thị ===
# Lưu đồ thị dưới dạng file Graph-tool
output_path = "social_network.gt"
g.save(output_path)
print(f"Đồ thị đã được lưu tại: {output_path}")
