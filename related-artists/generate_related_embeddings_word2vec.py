# DEPENDENCIES
import sys
import os
import networkx as nx
from gensim.models import Word2Vec

# MODULE DEPS
sys.path.append('../db')
sys.path.append('../node2vec/src')
from read_data_set import get_related_artists
import node2vec

# OUTPUT DIR
RESULT_DIR = './output'
if not os.path.exists(RESULT_DIR):
    os.mkdir(RESULT_DIR)
RESULT_FILE = os.path.join(RESULT_DIR, 'related_artist_word2vec_embeddings.emb')

NUM_WALKS = 10
WALK_LENGTH = 50
WEIGHT_PARAM_A = 1
WEIGHT_PARAM_B = 1
VECTOR_DIMENSION = 20
WINDOW_SIZE = 20
PARALLEL_WORKER_COUNT = 8
SGD_EPOCHS = 5


def get_nx_graph():
    edges = get_related_artists()
    G = nx.Graph()
    G.add_edges_from(edges)
    G = G.to_undirected()
    for edge in G.edges():
        G[edge[0]][edge[1]]['weight'] = 1
    G = G.to_undirected()
    return G


nx_G = get_nx_graph()
G = node2vec.Graph(nx_G, False, WEIGHT_PARAM_A, WEIGHT_PARAM_B)
G.preprocess_transition_probs()
walks = G.simulate_walks(NUM_WALKS, WALK_LENGTH)
model = Word2Vec(walks, size=VECTOR_DIMENSION, window=WINDOW_SIZE, min_count=0, sg=1, workers=PARALLEL_WORKER_COUNT, iter=SGD_EPOCHS)
word_vectors = model.wv
word_vectors.save_word2vec_format(RESULT_FILE)
