# STANDARD IMPORTS
import operator

# NON-STANDARD IMPORTS

# SKLEARN
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.svm import SVC
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Perceptron
from sklearn.pipeline import make_pipeline
import optunity
import optunity.metrics

# NUMPY
import numpy as np

# MODULE DEPS
from util import get_track_data

(track_ids, data, labels, user_names) = get_track_data(False, True, True, datasets=['spotify_toy_data_set'], transform_glove=True)
print(data)
index = 0
for i in data[:, 0]:
    if not np.isfinite(i):
        print(index, i)
    index += 1
svm = SVC(kernel='rbf', gamma='scale')
classifier = make_pipeline(SimpleImputer(), StandardScaler(), svm)
scores = cross_val_score(classifier, data, labels, cv=5)
result = 'RBF Kernel Scores: {}'.format(
    'Accuracy: %0.2f (+/- %0.2f)' % (scores.mean(), scores.std() * 2))
print(result)
