import numpy as np
from sklearn.preprocessing import StandardScaler
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

# MODULE DEPS
from read_data_set import *

THREED=False
NAME='myles'
# # GET DATA
# full_data = get_toy_set()
# full_data_arr = np.asarray(full_data['data_arr']).astype(np.float)
# labels = np.asarray(full_data['labels'])
#
# # GET GENRE DATA
# genre_only_data = get_toy_set_genre_only()
# genre_data_arr = np.asarray(genre_only_data['data_arr']).astype(np.float)

# GET USER DATA
full_data = get_user_set(NAME)
full_data_arr = np.asarray(full_data['data_arr']).astype(np.float)
labels = np.asarray(full_data['labels'])

# NORMALIZE DATA
# norm_data = StandardScaler().fit_transform(full_data_arr)
norm_data = StandardScaler().fit_transform(full_data_arr)

# PCA DIM REDUCTION
C = np.cov(np.transpose(norm_data))
evals, evecs = np.linalg.eigh(C)
V = evecs[:,(-4):-1] if THREED else evecs[:,(-3):-1]
reduced_data = norm_data @ V

# EXTRACTING CLASSES
data = []
for i in range(len(set(labels))):
    indices = [j for j in range(len(labels)) if labels[j] == i]
    d = (reduced_data[indices,0], reduced_data[indices,1], reduced_data[indices,2]) if THREED else (reduced_data[indices,0], reduced_data[indices,1])
    # d = (reduced_data[indices,0], reduced_data[indices,1])
    data.append(d)

cm = plt.get_cmap('gist_rainbow')
colors = [cm(1.5*i/float(len(set(labels)))) for i in range(len(set(labels)))]

# PLOTTING
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1, projection='3d') if THREED else fig.add_subplot(1, 1, 1)
for data, color in zip(data, colors):
    if THREED:
        x, y, z = data
        ax.scatter(x, y, z, alpha=0.8, c=color, s=100)
    else:
        x, y = data
        ax.scatter(x, y, alpha=0.8, c=color, s=100)

plt.legend(loc='best')
plt.savefig('results/{}data.png'.format(NAME))
plt.show()
