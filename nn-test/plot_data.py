import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

# MODULE DEPS
from read_data_set import get_toy_set

# GET DATA
full_data = get_toy_set()
full_data_arr = np.asarray(full_data['data_arr']).astype(np.float)
labels = np.asarray(full_data['labels'])

# NORMALIZE DATA
norm_data = StandardScaler().fit_transform(full_data_arr)

# PCA DIM REDUCTION
C = np.cov(np.transpose(norm_data))
evals, evecs = np.linalg.eigh(C)
V = evecs[:,-4:-1]
reduced_data = norm_data @ V

# EXTRACTING CLASSES
data = []
for i in range(len(set(labels))):
    indices = [j for j in range(len(labels)) if labels[j] == i]
    d = (reduced_data[indices,0],reduced_data[indices,1],reduced_data[indices,2])
    data.append(d)

cm = plt.get_cmap('gist_rainbow')
colors = [cm(2.*i/float(len(set(labels)))) for i in range(len(set(labels)))]

# PLOTTING
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1, projection='3d')

for data, color in zip(data, colors):
    x, y, z = data
    ax.scatter(x, y, z, alpha=0.5, c=color, edgecolors='none', s=30)

plt.title('Toy-Set Data')
plt.legend(loc=2)
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')
plt.show()
