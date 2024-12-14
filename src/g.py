from graph_tool.all import *

# Tạo đồ thị
g = Graph(directed=True)  # Đồ thị có hướng (directed graph)

# Thêm các thuộc tính cho đồ thị
user_map = g.new_vertex_property("string")  # Thuộc tính để lưu tên user
vertices = {}  # Map tên user -> vertex object

# Thêm các node và edges
for source, target in edges:
    if source not in vertices:
        v_source = g.add_vertex()
        user_map[v_source] = source
        vertices[source] = v_source
    if target not in vertices:
        v_target = g.add_vertex()
        user_map[v_target] = target
        vertices[target] = v_target
    g.add_edge(vertices[source], vertices[target])

print("Số lượng node:", g.num_vertices())
print("Số lượng cạnh:", g.num_edges())

# Tính toán layout để hiển thị đẹp
pos = sfdp_layout(g)  # Sử dụng lực đẩy (force-directed layout)

# Thêm thuộc tính cảm xúc cho node
sentiment = g.new_vertex_property("int")  # Thuộc tính lưu cảm xúc
for user, vertex in vertices.items():
    user_sentiment = data.loc[data["user"] == user, "target"].iloc[0]  # Lấy nhãn cảm xúc
    sentiment[vertex] = user_sentiment

# Vẽ đồ thị với màu sắc khác nhau cho cảm xúc
graph_draw(g, pos, vertex_fill_color=sentiment, vertex_text=user_map,
           output_size=(1000, 1000), output="emotion_propagation_colored.png")

