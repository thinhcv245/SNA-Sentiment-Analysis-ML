import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

print("Loading graph...")
G = nx.read_graphml("graph.graphml")

# Step 3: Define propagation function
def propagate_emotion(graph, alpha, steps):
    for node in graph.nodes:
        if 'emotion' not in graph.nodes[node]:
            graph.nodes[node]['emotion'] = 0
    positive_proportion = []

    for _ in range(steps):
        new_emotions = {}
        for node in graph.nodes:
            print(graph.nodes[node], graph.nodes[node]['emotion'], type(graph.nodes[node]['emotion']))
            if graph.nodes[node]['emotion'] > 0:
                continue

            # Influence from neighbors
            positive_neighbors = sum(
                1 for neighbor in graph.neighbors(node)
                if graph.nodes[neighbor]['emotion'] > 0
            )
            
            influence = alpha * positive_neighbors
            if influence > np.random.rand():
                new_emotions[node] = 1

        # Update emotions
        for node, emotion in new_emotions.items():
            graph.nodes[node]['emotion'] = emotion

        # Calculate positive proportion
        positive_count = sum(1 for _, data in graph.nodes(data=True) if data['emotion'] == 'positive')
        positive_proportion.append(positive_count / graph.number_of_nodes())

    return positive_proportion

# Step 4: Simulate for different alpha values
alphas = [0.1, 0.3, 0.5, 0.7, 1.0]
steps = 10
results = {}

for alpha in alphas:
    results[alpha] = propagate_emotion(G.copy(), alpha, steps)

# Step 5: Plot the results
plt.figure(figsize=(10, 6))
for alpha, proportions in results.items():
    plt.plot(range(steps), proportions, label=f'Alpha = {alpha}')

plt.xlabel('Steps')
plt.ylabel('Proportion of Positive Emotions')
plt.title('Effect of Activation Intensity (Alpha) on Positive Emotion Spread')
plt.legend()
plt.grid()
plt.show()
