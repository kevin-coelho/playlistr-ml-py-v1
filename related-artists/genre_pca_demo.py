# DEPENDENCIES
from glove import Glove
from glove import Corpus
import sklearn.decomposition
import numpy as np
import pandas as pd
import argparse
import chalk
import sys
import os
import seaborn as sns
import matplotlib.pyplot as plt
import random
import adjustText

# OUTPUT DIR
CWD = os.path.dirname(os.path.realpath(__file__))
RESULT_DIR = os.path.join(CWD, 'output')
CORPUS_FILE = os.path.join(RESULT_DIR, 'related_genre_glove_corpus.model')
GLOVE_MODEL_FILE = os.path.join(RESULT_DIR, 'related_genre_glove.model')

parser = argparse.ArgumentParser(description='Related genres PCA demo')
parser.add_argument('QUERY', action='store', default='', help='Demo PCA using this genre')
parser.add_argument('--corpus', '-c', default=CORPUS_FILE, help='Specify corpus file to read')
parser.add_argument('--glove', '-g', default=GLOVE_MODEL_FILE, help='Specify glove model file to read')
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
    print('[{}] Error reading corpus file.'.format(chalk.red(CORPUS_FILE)))
    quit(0)

if os.path.exists(GLOVE_MODEL_FILE):
    print('[{}] Reading glove model from file...'.format(chalk.yellow(GLOVE_MODEL_FILE)))
    glove = Glove.load(GLOVE_MODEL_FILE)
else:
    print('[{}] Error reading glove file.'.format(chalk.red(GLOVE_MODEL_FILE)))
    quit(0)

matrix = glove.word_vectors
dictionary = glove.dictionary

if args.QUERY not in dictionary:
    print('Artist name not found in dictionary. Try searching the db.')
    quit(0)

# get 3 close to our query
primary = matrix[dictionary[args.QUERY]]
similar = glove.most_similar(args.QUERY, number=4)
similar_labels = [elem[0] for elem in similar]
similar_data = matrix[[dictionary[item] for item in similar_labels]]
similar_colors = [args.QUERY for elem in similar]

# get a 2nd item randomly
second_name, second_idx = ('country', dictionary['country'])
while second_name == args.QUERY:
    second_name, second_idx = random.choice(list(dictionary.items()))
second = matrix[second_idx]
second_similar = glove.most_similar(second_name, number=4)
second_similar_labels = [elem[0] for elem in second_similar]
second_similar_data = matrix[[dictionary[item] for item in second_similar_labels]]
second_similar_colors = ['country' for elem in similar]

# get a 3rd item randomly
third_name, third_idx = ('salsa', dictionary['salsa'])
while third_name == args.QUERY:
    third_name, third_idx = random.choice(list(dictionary.items()))
third = matrix[third_idx]
third_similar = glove.most_similar(third_name, number=4)
third_similar_labels = [elem[0] for elem in third_similar]
third_similar_data = matrix[[dictionary[item] for item in third_similar_labels]]
third_similar_colors = ['salsa' for elem in similar]

X = np.array([primary, second, third])
for data in [similar_data, second_similar_data, third_similar_data]:
    X = np.append(X, data, axis=0)
labels = np.array([args.QUERY, second_name, third_name])
for data in [similar_labels, second_similar_labels, third_similar_labels]:
    labels = np.append(labels, data, axis=0)
colors = np.array([args.QUERY, 'country', 'salsa'])
for color_arr in [similar_colors, second_similar_colors, third_similar_colors]:
    colors = np.append(colors, color_arr, axis=0)

pca = sklearn.decomposition.PCA(n_components=2)
transformed = pca.fit_transform(X)
dataset = pd.DataFrame({'X': transformed[:, 0], 'Y': transformed[:, 1], 'color': colors})

ax = sns.lmplot(x='X', y='Y', hue='color', data=dataset, fit_reg=False, height=8, aspect=2.5, scatter_kws={'s': 300}, legend=False)
plt.title('Related genres: {}, country, salsa'.format(args.QUERY))
plt.xlabel('X')
plt.ylabel('Y')
texts = []


def label_point(ax):
    for (point, label) in zip(transformed, labels):
        x, y = point
        texts.append(ax.text(x + .02, y, str(label), fontsize=15))


label_point(plt.gca())
plt.legend(loc='center left')
plt.subplots_adjust(right=1)
plt.tight_layout(pad=2)
adjustText.adjust_text(texts)
plt.show()

"""
#print(glove.word_vectors[glove.dictionary[args.query]])
#print(glove.most_similar(args.query, number=40))
"""