# STANDARD IMPORTS

# NON-STANDARD IMPORTS

# SKLEARN 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Perceptron


# NUMPY
import numpy as np

# MODULE DEPS
from read_data_set import get_toy_set

data = get_toy_set()
data_arr = data['data_arr']
labels = data['labels']

# SPLIT DATA
x_train, x_test, y_train, y_test = train_test_split(data_arr, labels, test_size=0.30, random_state=0)
sc_X = StandardScaler()
x_train = sc_X.fit_transform(x_train)
x_test = sc_X.transform(x_test)

# TRAIN CLASSIFIERS
rbf_classifier = SVC(kernel='rbf', gamma='scale')
rbf_classifier.fit(x_train, y_train)
rbf_results = rbf_classifier.predict(x_test)
rbf_accuracy = 1 - (np.sum([1 for idx, result in enumerate(rbf_results) if result == y_test[idx]]) / y_test.shape[0])
print('RBF Accuracy: {}'.format(rbf_accuracy))

sigmoid_classifier = SVC(kernel='sigmoid', gamma='scale')
sigmoid_classifier.fit(x_train, y_train)
sigmoid_results = sigmoid_classifier.predict(x_test)
sigmoid_accuracy = 1 - (np.sum([1 for idx, result in enumerate(sigmoid_results) if result == y_test[idx]]) / y_test.shape[0])
print('Sigmoid Accuracy: {}'.format(sigmoid_accuracy))

#x_train_poly = PolynomialFeatures(degree=2).fit_transform(x_train)
#x_test_poly = PolynomialFeatures(degree=2).fit_transform(x_test)

clf = Perceptron(fit_intercept=False, tol=None, shuffle=False).fit(x_train, y_train)
clf_results = clf.predict(x_test)
clf_accuracy = 1 - (np.sum([1 for idx, result in enumerate(clf_results) if result == y_test[idx]]) / y_test.shape[0])
print('CLF Accuracy: {}'.format(clf_accuracy))

poly_classifier = SVC(kernel='poly', degree=4, gamma='scale')
poly_classifier.fit(x_train, y_train)
poly_results = poly_classifier.predict(x_test)
poly_accuracy = 1 - (np.sum([1 for idx, result in enumerate(poly_results) if result == y_test[idx]]) / y_test.shape[0])
print('POLY Accuracy: {}'.format(poly_accuracy))
