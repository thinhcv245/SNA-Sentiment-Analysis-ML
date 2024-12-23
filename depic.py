import networkx as nx

G = nx.read_graphml("graph.graphml")

print(G.number_of_nodes(), G.number_of_edges())