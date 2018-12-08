# DEPENDENCIES
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import sys

# MODULE DEPS
sys.path.append('../db')
from read_data_set import get_related_artists

G = nx.Graph()
edges = get_related_artists()
print(edges[0])
G.add_edges_from(edges[0:10000])
G = nx.convert_node_labels_to_integers(G)
plt.subplot(121)
nx.draw(G, with_labels=False, **{'node_size': 5})
plt.show()
