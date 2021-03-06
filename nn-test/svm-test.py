# STANDARD IMPORTS
import operator

# NON-STANDARD IMPORTS

# SKLEARN
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
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


# PARAMETERS TO TUNE
# C, gamma, coef0
"""
decision_function_shape : ‘ovo’, ‘ovr’, default=’ovr’
Whether to return a one-vs-rest (‘ovr’) decision function of shape (n_samples, n_classes) as all other classifiers, or the original one-vs-one (‘ovo’) decision function of libsvm which has shape (n_samples, n_classes * (n_classes - 1) / 2). However, one-vs-one (‘ovo’) is always used as multi-class strategy.

"""
def run_sigmoid_kernel(scale_features=False, genre_only=False):
    data = genre_data_arr if genre_only else full_data_arr
    svm = SVC(kernel='sigmoid', gamma='scale')
    classifier = make_pipeline(StandardScaler(), svm) if scale_features else svm
    scores = cross_val_score(classifier, data, labels, cv=5)
    result = '[{}] [{}] Sigmoid Kernel Scores: {}'.format(
        'Genre Only' if genre_only else 'Full',
        'Scaled' if scale_features else 'Unscaled',
        'Accuracy: %0.2f (+/- %0.2f)' % (scores.mean(), scores.std() * 2))
    return result


def run_poly_kernel(scale_features=False, genre_only=False):
    data = genre_data_arr if genre_only else full_data_arr
    svm = SVC(kernel='poly', gamma='scale', degree=4)
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


def run_logistic(scale_features=False, genre_only=False):
    data = genre_data_arr if genre_only else full_data_arr
    logist = LogisticRegression()
    classifier = make_pipeline(StandardScaler(), logist) if scale_features else logist
    scores = cross_val_score(classifier, data, labels, cv=5)
    result = '[{}] [{}] Logistic Regression Scores: {}'.format(
        'Genre Only' if genre_only else 'Full',
        'Scaled' if scale_features else 'Unscaled',
        'Accuracy: %0.2f (+/- %0.2f)' % (scores.mean(), scores.std() * 2))
    return result


results = []
for scale in [True, False]:
    for genre_only in [True, False]:
            results.append(run_logistic(scale, genre_only))
print('\n###### INITIAL RESULTS ######\n\t' + '\n\t'.join(results))

"""
results = []
for scale in [True, False]:
    for genre_only in [True, False]:
        results.append(run_rbf_kernel(scale, genre_only))
        results.append(run_sigmoid_kernel(scale, genre_only))
        results.append(run_poly_kernel(scale, genre_only))
        results.append(run_perceptron(scale, genre_only))
print('\n###### INITIAL RESULTS ######\n\t' + '\n\t'.join(results))"""

"""
scaler = StandardScaler()
scaled_full_data = scaler.fit_transform(full_data_arr)

c_vals = {}

# score function: twice iterated 10-fold cross-validated accuracy
@optunity.cross_validated(x=scaled_full_data, y=labels, num_folds=5, num_iter=2)
def svm_auc(x_train, y_train, x_test, y_test, logC):
    model = SVC(kernel='sigmoid', C=10 ** logC, gamma='scale').fit(x_train, y_train)
    accuracy = model.score(x_test, y_test)
    print('tuning...', logC, accuracy)
    decision_values = model.decision_function(x_test)
    if str(logC) not in c_vals:
        c_vals[str(logC)] = accuracy
    else:
        c_vals[str(logC)] += accuracy / 10
    return accuracy
    # return optunity.metrics.roc_auc(y_test, decision_values)


# perform tuning
hps, _, _ = optunity.maximize(svm_auc, num_evals=25, logC=[-5, 2])

# print('best c: {}'.format(max(c_vals.items(), key=operator.itemgetter(1))[0]))

# train model on the full training set with tuned hyperparameters
optimal_model = SVC(kernel='sigmoid', C=10 ** hps['logC'], gamma='scale')
# optimal_model = SVC(kernel='sigmoid', C=16.49, gamma='scale')
scores = cross_val_score(optimal_model, scaled_full_data, labels, cv=5)
result = 'Optimal Model Scores (C={}): {}'.format(10 ** hps['logC'], 'Accuracy: %0.2f (+/- %0.2f)' % (scores.mean(), scores.std() * 2))
print(result)
"""
