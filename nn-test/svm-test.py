# STANDARD IMPORTS

# NON-STANDARD IMPORTS

# SKLEARN 
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Perceptron
from sklearn.pipeline import make_pipeline

# NUMPY
import numpy as np

# MODULE DEPS
from read_data_set import get_toy_set, get_toy_set_genre_only

# GET DATA
full_data = get_toy_set()
full_data_arr = full_data['data_arr']
labels = full_data['labels']

genre_only_data = get_toy_set_genre_only()
genre_data_arr = genre_only_data['data_arr']


def run_rbf_kernel(scale_features=False, genre_only=False):
    data = genre_data_arr if genre_only else full_data_arr
    svm = SVC(kernel='rbf', gamma='scale')
    classifier = make_pipeline(StandardScaler(), svm) if scale_features else svm
    scores = cross_val_score(classifier, data, labels, cv=5)
    result = '[{}] [{}] RBF Kernel Scores: {}'.format(
        'Genre Only' if genre_only else 'Full',
        'Scaled' if scale_features else 'Unscaled',
        'Accuracy: %0.2f (+/- %0.2f)' % (scores.mean(), scores.std() * 2))
    return result


def run_sigmoid_kernel(scale_features=False, genre_only=False):
    data = genre_data_arr if genre_only else full_data_arr
    svm = SVC(kernel='rbf', gamma='scale')
    classifier = make_pipeline(StandardScaler(), svm) if scale_features else svm
    scores = cross_val_score(classifier, data, labels, cv=5)
    result = '[{}] [{}] Sigmoid Kernel Scores: {}'.format(
        'Genre Only' if genre_only else 'Full',
        'Scaled' if scale_features else 'Unscaled',
        'Accuracy: %0.2f (+/- %0.2f)' % (scores.mean(), scores.std() * 2))
    return result


def run_poly_kernel(degree, scale_features=False, genre_only=False):
    data = genre_data_arr if genre_only else full_data_arr
    svm = SVC(kernel='rbf', gamma='scale')
    classifier = make_pipeline(StandardScaler(), svm) if scale_features else svm
    scores = cross_val_score(classifier, data, labels, cv=5)
    result = '[{}] [{}] Poly Kernel Scores: {}'.format(
        'Genre Only' if genre_only else 'Full',
        'Scaled' if scale_features else 'Unscaled',
        'Accuracy: %0.2f (+/- %0.2f)' % (scores.mean(), scores.std() * 2))
    return result


def run_perceptron(scale_features=False, genre_only=False):
    data = genre_data_arr if genre_only else full_data_arr
    percept = Perceptron(fit_intercept=False, max_iter=1000, tol=1e-3, shuffle=True)
    classifier = make_pipeline(StandardScaler(), percept) if scale_features else percept
    scores = cross_val_score(classifier, data, labels, cv=5)
    result = '[{}] [{}] Perceptron Scores: {}'.format(
        'Genre Only' if genre_only else 'Full',
        'Scaled' if scale_features else 'Unscaled',
        'Accuracy: %0.2f (+/- %0.2f)' % (scores.mean(), scores.std() * 2))
    return result


results = []
for scale in [True, False]:
    for genre_only in [True, False]:
        results.append(run_rbf_kernel(scale, genre_only))
        results.append(run_sigmoid_kernel(scale, genre_only))
        results.append(run_poly_kernel(scale, genre_only))
        results.append(run_perceptron(scale, genre_only))
print('\n###### RESULTS ######\n\t' + '\n\t'.join(results))
