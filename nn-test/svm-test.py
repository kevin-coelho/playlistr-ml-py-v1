# STANDARD IMPORTS

# NON-STANDARD IMPORTS

# SKLEARN
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Perceptron


# NUMPY
import numpy as np

# MODULE DEPS
from read_data_set import get_toy_set, get_toy_set_genre_only

# GET DATA
data = get_toy_set()
data_arr = data['data_arr']
labels = data['labels']

genre_only_data = get_toy_set_genre_only()
genre_data_arr = genre_only_data['data_arr']

# SPLIT DATA
# kf = KFold(n_splits=4, shuffle=True)
# kf.get_n_splits(data_arr)

x_train, x_test, y_train, y_test = train_test_split(data_arr, labels, test_size=0.30)
sc_X = StandardScaler()
x_train = sc_X.fit_transform(x_train)
x_test = sc_X.transform(x_test)

# kf_g = KFold(n_splits=4, shuffle=True)
# kf_g.get_n_splits(genre_data_arr)
genre_only_x_train, genre_only_x_test, genre_only_y_train, genre_only_y_test = train_test_split(genre_data_arr, labels, test_size=0.30)

# for train_index, test_index in kf.split(data_arr):

# TRAIN CLASSIFIERS
rbf_classifier = SVC(kernel='rbf', gamma='scale')
rbf_classifier.fit(x_train, y_train)
rbf_results = rbf_classifier.predict(x_test)
rbf_accuracy = (np.sum([1 for idx, result in enumerate(rbf_results) if result == y_test[idx]]) / y_test.shape[0])
print('RBF Kernel Accuracy: {}'.format(rbf_accuracy))

sigmoid_classifier = SVC(kernel='sigmoid', gamma='scale')
sigmoid_classifier.fit(x_train, y_train)
sigmoid_results = sigmoid_classifier.predict(x_test)
sigmoid_accuracy = (np.sum([1 for idx, result in enumerate(sigmoid_results) if result == y_test[idx]]) / y_test.shape[0])
print('Sigmoid Kernel Accuracy: {}'.format(sigmoid_accuracy))

# x_train_poly = PolynomialFeatures(degree=2).fit_transform(x_train)
# x_test_poly = PolynomialFeatures(degree=2).fit_transform(x_test)

percept = Perceptron(fit_intercept=False, max_iter=1000, tol=1e-3, shuffle=False).fit(x_train, y_train)
percept_results = percept.predict(x_test)
percept_accuracy = (np.sum([1 for idx, result in enumerate(percept_results) if result == y_test[idx]]) / y_test.shape[0])
print('Perceptron Accuracy: {}'.format(percept_accuracy))

poly_classifier = SVC(kernel='poly', degree=4, gamma='scale')
poly_classifier.fit(x_train, y_train)
poly_results = poly_classifier.predict(x_test)
poly_accuracy = (np.sum([1 for idx, result in enumerate(poly_results) if result == y_test[idx]]) / y_test.shape[0])
print('POLY Kernel Accuracy: {}'.format(poly_accuracy))

# for train_index, test_index in kf.split(genre_data_arr):

print('\n##### GENRE ONLY #####\n')
genre_poly_classifier = SVC(kernel='poly', degree=4, gamma='scale')
genre_poly_classifier.fit(genre_only_x_train, genre_only_y_train)
genre_poly_results = genre_poly_classifier.predict(genre_only_x_test)
genre_poly_accuracy = (np.sum([1 for idx, result in enumerate(genre_poly_results) if result == genre_only_y_test[idx]]) / genre_only_y_test.shape[0])
print('POLY Kernel Accuracy: {}'.format(genre_poly_accuracy))

genre_percept = Perceptron(fit_intercept=False, max_iter=1000, tol=1e-3, shuffle=False).fit(genre_only_x_train, genre_only_y_train)
genre_percept_results = genre_percept.predict(genre_only_x_test)
genre_percept_accuracy = (np.sum([1 for idx, result in enumerate(genre_percept_results) if result == genre_only_y_test[idx]]) / genre_only_y_test.shape[0])
print('Perceptron Accuracy: {}'.format(genre_percept_accuracy))
