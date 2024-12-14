import pandas as pd
import re

# Load Sentiment140 dataset
data = pd.read_csv("data/raw/dataset_base.csv", encoding='latin-1', header=None)
data.columns = ["target", "id", "date", "flag", "user", "text"]

# Chỉ giữ các cột cần thiết
data = data[["target", "user", "text"]]

# Hàm trích xuất mentions và retweets từ nội dung tweet
def extract_mentions(text):
    mentions = re.findall(r"@\w+", text)  # Lấy danh sách các user được tag (@user)
    return mentions

def is_retweet(text):
    return re.match(r"^RT @\w+:", text) is not None  # Xác định tweet có phải là retweet không

# Áp dụng xử lý lên cột text
data["mentions"] = data["text"].apply(extract_mentions)
data["is_retweet"] = data["text"].apply(is_retweet)

print(data.head())

from collections import defaultdict

# Tạo danh sách các node và edges
edges = []
for _, row in data.iterrows():
    user = row["user"]
    mentions = row["mentions"]
    if row["is_retweet"] and mentions:  # Xử lý retweet
        edges.append((user, mentions[0]))  # Retweet tạo kết nối
    for mentioned_user in mentions:  # Xử lý mentions
        edges.append((user, mentioned_user))

print("Số lượng cạnh:", len(edges))

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


