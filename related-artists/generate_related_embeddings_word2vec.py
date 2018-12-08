# DEPENDENCIES
import sys
import os
from gensim.models import Word2Vec
import multiprocessing

# MODULE DEPS
from util import get_nx_graph, get_node2vec_walks

# OUTPUT DIR
RESULT_DIR = './output'
if not os.path.exists(RESULT_DIR):
    os.mkdir(RESULT_DIR)
RESULT_FILE = os.path.join(RESULT_DIR, 'related_artist_word2vec_embeddings.emb')

# CONSTANTS
VECTOR_DIMENSION = 20
WINDOW_SIZE = 20
PARALLEL_WORKER_COUNT = multiprocessing.cpu_count()
SGD_EPOCHS = 5

# MAIN
nx_G = get_nx_graph()
walks = get_node2vec_walks(nx_G)
model = Word2Vec(walks, size=VECTOR_DIMENSION, window=WINDOW_SIZE, min_count=0, sg=1, workers=PARALLEL_WORKER_COUNT, iter=SGD_EPOCHS)
word_vectors = model.wv
word_vectors.save_word2vec_format(RESULT_FILE)
