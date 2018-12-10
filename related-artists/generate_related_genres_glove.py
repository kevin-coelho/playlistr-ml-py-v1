# DEPENDENCIES
import sys
import os
import networkx as nx
from glove import Glove
from glove import Corpus
import multiprocessing
import argparse
import chalk

# MODULE DEPS
import util

# OUTPUT DIR
CWD = os.path.dirname(os.path.realpath(__file__))
RESULT_DIR = os.path.join(CWD, 'output')
CORPUS_FILE = os.path.join(RESULT_DIR, 'related_genre_glove_corpus.model')
GLOVE_MODEL_FILE = os.path.join(RESULT_DIR, 'related_genre_glove.model')

# CONSTANTS
WINDOW_SIZE = 5
VECTOR_DIMENSION = 128
GLOVE_EPOCHS = 15
PARALLEL_WORKER_COUNT = multiprocessing.cpu_count()

parser = argparse.ArgumentParser(description='Related genres GLOVE model')
parser.add_argument('--corpus', '-c', default=CORPUS_FILE, help='Specify corpus file to read')
parser.add_argument('--glove', '-g', default=GLOVE_MODEL_FILE, help='Specify glove model file to read')
parser.add_argument('--train', '-t', action='store_true', default=False, help='Retrain glove model from corpus')
parser.add_argument('--query', '-q', action='store', default='', help='Get close genres')
args = parser.parse_args()

CORPUS_FILE = args.corpus
GLOVE_MODEL_FILE = args.glove

if not os.path.exists(RESULT_DIR):
    os.mkdir(RESULT_DIR)

# MAIN
if os.path.exists(CORPUS_FILE):
    print('[{}] Reading corpus from file...'.format(chalk.yellow(CORPUS_FILE)))
    corpus = Corpus.load(CORPUS_FILE)
else:
    nx_G = util.get_nx_graph()
    walks = util.get_node2vec_walks(nx_G)
    corpus = Corpus()
    corpus.fit(walks, window=WINDOW_SIZE)
    print('[{}] Writing corpus file...'.format(chalk.green(CORPUS_FILE)))
    corpus.save(CORPUS_FILE)

if os.path.exists(GLOVE_MODEL_FILE) and not args.train:
    print('[{}] Reading glove model from file...'.format(chalk.yellow(GLOVE_MODEL_FILE)))
    glove = Glove.load(GLOVE_MODEL_FILE)
else:
    glove = Glove(no_components=VECTOR_DIMENSION, learning_rate=0.05)
    glove.fit(corpus.matrix, epochs=GLOVE_EPOCHS, no_threads=PARALLEL_WORKER_COUNT, verbose=True)
    glove.add_dictionary(corpus.dictionary)
    print('[{}] Writing glove file...'.format(chalk.green(GLOVE_MODEL_FILE)))
    glove.save(GLOVE_MODEL_FILE)
if args.query:
    dictionary = glove.dictionary
    print(glove.word_vectors[glove.dictionary[args.query]])
    print(glove.most_similar(args.query, number=10))


def get_glove_model():
    return glove
