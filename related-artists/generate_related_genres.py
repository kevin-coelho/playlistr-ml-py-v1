# DEPENDENCIES
import sys
import os
import networkx as nx
from glove import Glove
from glove import Corpus
import multiprocessing
import numpy as np
import chalk

# MODULE DEPS
from util import related_genres

# OUTPUT DIR
CWD = os.path.dirname(os.path.realpath(__file__))
RESULT_DIR = os.path.join(CWD, 'output')
if not os.path.exists(RESULT_DIR):
    os.mkdir(RESULT_DIR)
CORPUS_FILE = os.path.join(RESULT_DIR, 'related_genre_glove_corpus.model')
GLOVE_MODEL_FILE = os.path.join(RESULT_DIR, 'related_genre_glove.model')

# CONSTANTS
WINDOW_SIZE = 1
VECTOR_DIMENSION = 128
GLOVE_EPOCHS = 100
PARALLEL_WORKER_COUNT = multiprocessing.cpu_count()

# MAIN
res = related_genres()
print('Got related genres: ', chalk.green(len(res)))
artistIds, genres = zip(*res)
corpus = Corpus()
corpus.fit(genres, window=WINDOW_SIZE)
corpus.save(CORPUS_FILE)

glove = Glove(no_components=VECTOR_DIMENSION, learning_rate=0.05)
glove.fit(corpus.matrix, epochs=GLOVE_EPOCHS, no_threads=PARALLEL_WORKER_COUNT, verbose=False)
glove.add_dictionary(corpus.dictionary)
glove.save(GLOVE_MODEL_FILE)
for genre in genres[0: 50]:
    print(genre[0], ':', glove.most_similar(genre[0]))
