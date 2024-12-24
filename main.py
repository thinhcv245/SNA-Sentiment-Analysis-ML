import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import random

# Load Sentiment140 Dataset
def load_data(file_path):
    # Load dataset with appropriate column names
    data = pd.read_csv(file_path, encoding='latin-1', names=['target', 'id', 'date', 'query', 'user', 'text'])
    # Map target values to emotions (-1: Negative, 0: Neutral, 1: Positive)
    data['emotion'] = data['target'].map({0: -1, 2: 0, 4: 1})
    return data

# Build Social Network Graph
def build_graph(data):
    G = nx.DiGraph()
    for _, row in data.iterrows():
        user = row['user']
        mentions = [mention.strip('@') for mention in row['text'].split() if mention.startswith('@')]
        if not G.has_node(user):
            G.add_node(user)

        # Thêm các cạnh từ user đến mentions và tăng trọng số nếu đã tồn tại
        for mention in mentions:
            if not G.has_node(mention):
                G.add_node(mention)  # Đảm bảo mention cũng là node
            if not G.has_edge(user, mention):
                G.add_edge(user, mention, weight=1)
            else:
                G[user][mention]['weight'] += 1

        # Gán thuộc tính emotion cho user
        G.nodes[user]['emotion'] = row['emotion']
    return G

# Propagation Model
def propagate_emotion(G, alpha=0.9, beta=0.5, lambda_=0.1, steps=10):
    for _ in range(steps):
        updated_emotions = {}
        for node in G.nodes():
            current_emotion = G.nodes[node].get('emotion', 0)
            neighbors = list(G.neighbors(node))
            for neighbor in neighbors:
                weight = G[node][neighbor].get('weight', 1)
                prob = beta * (1 - (1 / (1 + lambda_ * weight)))
                if random.random() < prob:
                    if neighbor not in updated_emotions:
                        updated_emotions[neighbor] = 0
                    updated_emotions[neighbor] += alpha * current_emotion
        for node, new_emotion in updated_emotions.items():
            G.nodes[node]['emotion'] = new_emotion
    return G

# Analyze the Graph
def analyze_graph(G):
    # Calculate density and clustering coefficient
    density = nx.density(G)
    clustering = nx.average_clustering(G.to_undirected())

    # Calculate emotion distribution
    emotions = [G.nodes[node].get('emotion', 0) for node in G.nodes()]
    emotion_distribution = {
        'negative': sum(1 for e in emotions if e < 0),
        'neutral': sum(1 for e in emotions if e == 0),
        'positive': sum(1 for e in emotions if e > 0),
    }
    return density, clustering, emotion_distribution

# Visualize the Graph
def visualize_graph(G):
    pos = nx.spring_layout(G)
    emotions = nx.get_node_attributes(G, 'emotion')
    colors = ["red" if emotions[node] < 0 else "green" if emotions[node] > 0 else "blue" for node in G.nodes()]

    plt.figure(figsize=(12, 8))
    nx.draw(G, pos, with_labels=False, node_color=colors, node_size=50, edge_color="gray", alpha=0.7)
    plt.title("Emotion Propagation in Social Network")
    plt.show()

def print_analys(density, clustering, emotion_distribution):
    from tabulate import tabulate
    data = [
        ["Graph Density", density],
        ["Average Clustering Coefficient", clustering],
        ["Emotion Distribution", emotion_distribution]
    ]
    table = tabulate(data, headers=["Metric", "Value"], tablefmt="pretty")
    print(table)

# Main Function
def main():
    # File path to the Sentiment140 dataset
    file_path = 'data/sentiment140.csv'  # Replace with your file path

    # print("Loading data...")
    # data = load_data(file_path)

    # print("Building graph...")
    # G = build_graph(data)
    # nx.write_graphml(G, "graph.graphml")
    # return
    print("Loading graph...")
    G = nx.read_graphml("graph.graphml")
    density, clustering, emotion_distribution = analyze_graph(G)
    print_analys(density=density, clustering=clustering, emotion_distribution=emotion_distribution)
    
    print("Running emotion propagation model...")
    G = propagate_emotion(G, alpha=0.9, beta=0.5, lambda_=0.1, steps=10)
    
    print("Analyzing graph...")
    density, clustering, emotion_distribution = analyze_graph(G)
    print_analys(density=density, clustering=clustering, emotion_distribution=emotion_distribution)

    return
    print("Visualizing graph...")
    visualize_graph(G)

if __name__ == "__main__":
    main()
