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

# Tạo danh sách các user duy nhất từ dataset
user_target_map = {user: target for user, target in zip(data["user"], data["target"])}
# Tạo danh sách các node và edges
edges = {}  # Lưu trữ các cạnh dưới dạng (source, target): weight
user_set = set()  # Set chứa tất cả các user duy nhất
for _, row in data.iterrows():
    user = row["user"]
    mentions = row["mentions"]
    if row["is_retweet"] and mentions:  # Xử lý retweet
        mention = mentions[0].replace("@", "")
        edge = (user, mention)
        if edge in edges:
            edges[edge] += 1
        else:
            edges[edge] = 1
        user_set.add(mention)
    for mentioned_user in mentions:  # Xử lý mentions
        mention = mentioned_user.replace("@", "")
        edge = (user, mention)
        if edge in edges:
            edges[edge] += 1
        else:
            edges[edge] = 1
        user_set.add(mention)
    user_set.add(user)  # Thêm user vào set

# Ghi các node vào file CSV, gán target = 0 nếu user không tồn tại trong dataset
nodes = []
for user in user_set:
    target = user_target_map.get(user, -1)  # Nếu user không có trong dataset, gán target = 0
    nodes.append((user, target))

nodes_df = pd.DataFrame(nodes, columns=["Label", "Setiment"])
nodes_df["Id"] = range(1, len(nodes_df) + 1)
nodes_df = nodes_df[["Id", "Label", "Setiment"]]
print(f"Số lượng nodes: {len(nodes)}")
nodes_df.to_csv("nodes.csv", index=False)

edges_list = [(source, target, weight) for (source, target), weight in edges.items()]
edges_df = pd.DataFrame(edges_list, columns=["Source", "Target", "Weight"])
print(f"Số lượng edges: {len(edges)}")
edges_df.to_csv("edges.csv", index=False)