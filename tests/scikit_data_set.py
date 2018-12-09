# Kernel SVM

# Importing the libraries
import psycopg2
import numpy as np
# import matplotlib.pyplot as plt
import pandas as pd

# Importing the datasets

# read data from database
conn = psycopg2.connect("dbname='playlistr_ml_v1' user='playlistr_ml_v1' host='localhost' password='plt_210'")
cur = conn.cursor()

cur.execute(open('get_track_features.sql', 'r').read())
features = cur.fetchall()
features_array = np.array([row for row in features])
track_ids = features_array[:, 0]
X = features_array[:, 1:-1].astype(np.float)
Y = features_array[:, -1]
print(track_ids, X, Y)
print(track_ids.shape, X.shape, Y.shape)
# unique_tracks = set(track_ids)
playlist_to_int = {list(set(Y))[i]: i for i in range(len(list(set(Y))))}
for i in range(len(Y)):
    Y[i] = playlist_to_int[Y[i]]

# track_to_playlist = {row[1]: row[0] for row in playlist_array}

# track_to_int = {list(unique_tracks)[i]: i for i in range(len(list(unique_tracks)))}

# cur.execute(open('get_track_genres.sql', 'r').read())
# genres = cur.fetchall()
# genre_array = np.array([row for row in genres])
# track_genres = {}
# for row in genre_array:
#     if row[0] in track_genres.keys():
#         track_genres[row[0]].add(row[1])
#     else:
#         track_genres[row[0]] = set()
#         track_genres[row[0]].add(row[1])

# cur.execute(open('get_playlist_tracks.sql', 'r').read())
# playlists = cur.fetchall()
# playlist_array = np.array([row for row in playlists])

# for row in playlist_array:
    # if row[1] in track_to_playlist.keys():
    #     track_to_playlist[row[1]].add(row[0])
    # else:
    #     track_to_playlist[row[1]] = set()
    #     track_to_playlist[row[1]].add(row[0])

# Splitting the dataset into the Training set and Test set

from sklearn.model_selection import train_test_split
X_Train, X_Test, Y_Train, Y_Test = train_test_split(X, Y, test_size = 0.25, random_state = 0)
print(X_Train, X_Test, Y_Train, Y_Test)

# Feature Scaling

from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_Train = sc_X.fit_transform(X_Train)
X_Test = sc_X.transform(X_Test)

# Fitting the classifier into the Training set

from sklearn.svm import SVC
classifier = SVC(kernel = 'rbf', random_state = 0)

# Predicting the test set results

Y_Pred = classifier.predict(X_Test)

# Making the Confusion Matrix

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_Test, Y_Pred)

# Visualising the Training set results

# from matplotlib.colors import ListedColormap
# X_Set, Y_Set = X_Train, Y_Train
# X1, X2 = np.meshgrid(np.arange(start = X_Set[:, 0].min() - 1, stop = X_Set[:, 0].max() + 1, step = 0.01),
#                      np.arange(start = X_Set[:, 1].min() - 1, stop = X_Set[:, 1].max() + 1, step = 0.01))
# plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
#              alpha = 0.75, cmap = ListedColormap(('red', 'green')))
# plt.xlim(X1.min(), X1.max())
# plt.ylim(X2.min(), X2.max())
# for i, j in enumerate(np.unique(Y_Set)):
#     plt.scatter(X_Set[Y_Set == j, 0], X_Set[Y_Set == j, 1],
#                 c = ListedColormap(('red', 'green'))(i), label = j)
# plt.title('Kernel SVM (Training set)')
# plt.xlabel('Age')
# plt.ylabel('Estimated Salary')
# plt.legend()
# plt.show()

# Visualising the Test set results

# from matplotlib.colors import ListedColormap
# X_Set, Y_Set = X_Test, Y_Test
# X1, X2 = np.meshgrid(np.arange(start = X_Set[:, 0].min() - 1, stop = X_Set[:, 0].max() + 1, step = 0.01),
#                      np.arange(start = X_Set[:, 1].min() - 1, stop = X_Set[:, 1].max() + 1, step = 0.01))
# plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
#              alpha = 0.75, cmap = ListedColormap(('red', 'green')))
# plt.xlim(X1.min(), X1.max())
# plt.ylim(X2.min(), X2.max())
# for i, j in enumerate(np.unique(Y_Set)):
#     plt.scatter(X_Set[Y_Set == j, 0], X_Set[Y_Set == j, 1],
#                 c = ListedColormap(('red', 'green'))(i), label = j)
# plt.title('Kernel SVM (Test set)')
# plt.xlabel('Age')
# plt.ylabel('Estimated Salary')
# plt.legend()
# plt.show()
