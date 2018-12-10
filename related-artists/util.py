# DEPENDENCIES
import sys
import os
import networkx as nx

# MODULE DEPS
CWD = os.path.dirname(os.path.realpath(__file__))
DB_FOLDER = os.path.realpath(os.path.join(CWD, '../db'))
NODE2VEC_FOLDER = os.path.realpath(os.path.join(CWD, '../node2vec/src'))
sys.path.append(DB_FOLDER)
sys.path.append(NODE2VEC_FOLDER)
from read_data_set import get_related_artists, get_related_genres
import node2vec

NUM_WALKS = 10
WALK_LENGTH = 20
WEIGHT_PARAM_A = 1
WEIGHT_PARAM_B = 1


def related_genres():
    return get_related_genres()


def get_nx_graph():
    edges = get_related_artists()
    G = nx.Graph()
    G.add_edges_from(edges)
    G = G.to_undirected()
    for edge in G.edges():
        G[edge[0]][edge[1]]['weight'] = 1
    G = G.to_undirected()
    return G


def get_node2vec_walks(nx_G):
    G = node2vec.Graph(nx_G, False, WEIGHT_PARAM_A, WEIGHT_PARAM_B)
    G.preprocess_transition_probs()
    walks = G.simulate_walks(NUM_WALKS, WALK_LENGTH)
    return walks
