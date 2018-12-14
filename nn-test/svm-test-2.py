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


def run_rbf(name, data, labels, C=1):
    print('[{}] Running set... shape: {}'.format(name, data.shape))
    svm = SVC(kernel='rbf', gamma='scale', class_weight='balanced', C=C)
    classifier = make_pipeline(StandardScaler(), svm)
    scores = cross_val_score(classifier, data, labels, cv=5)
    result = 'RBF Kernel Scores: {}'.format(
        'Accuracy: %0.2f (+/- %0.2f)' % (scores.mean(), scores.std() * 2))
    # print(result)
    return scores.mean()


sets = [
    ['Work Playlist', '\'\'Nam', 'domesticated antelope', 'Silicon', 'Germanium', 'Iridium', 'Silver'],
    ['Gold', 'Sodium', 'Cobalt', 'Vous parlez Francaise?', 'Tejas'],
    ['Clueless Vol 3', 'Club can\'\'t Handel', 'Free, Wild, and Young', 'Titan Fall Noob'],
    ['Wild, Free, and Young', 'Clueless in this Genre', 'DISCOTHEQUE 1996'],
    ['Julianne\'\'s V-Day HW Mixtape', 'Loud AF', 'Fresno Fuccboi', 'Road trips and other itineraries'],
    ['Alan for President', 'Princess Feel Good', 'Dance Girl X-treme', 'Baggheim + Sissyfus', 'The good shit', 'The Demagorgon'],
    ['Snow Ball', 'THE MOTHER FUCKING BLENDER', 'Starlord_44', 'Starlord_24'],

]


all_sets = np.array([item for sublist in sets for item in sublist])
print(all_sets)
average = 0.0
for idx in range(100):
    print(str(idx) + '\n')
    np.random.shuffle(all_sets)
    playlists = list(all_sets[0:5])
    track_ids, data, labels, user_names, track_dict, playlist_dict = get_track_data(audio_features=True, genres=True, artists=True, playlists=playlists, transform_glove=True)
    average += run_rbf('Miz Set {}'.format(idx), data, labels, C=3)/100.0

print("Miz average test accuracy: ", average)


# idx = 4
# track_ids, data, labels, user_names, track_dict, playlist_dict = get_track_data(audio_features=True, genres=True, artists=True, playlists=sets[idx], transform_glove=True)
# run_rbf('Miz Set {}'.format(idx), data, labels, C=2)


"""
for idx, miz_set in enumerate(sets):
    track_ids, data, labels, user_names, track_dict, playlist_dict = get_track_data(audio_features=True, genres=True, artists=True, playlists=miz_set, transform_glove=True)
    run_rbf('Miz Set {}'.format(idx), data, labels, C=3)"""

"""
track_ids, data, labels, user_names, track_dict, playlist_dict = get_track_data(audio_features=True, genres=True, artists=True, datasets=['spotify_toy_data_set'], transform_glove=True)
print(data.shape)
svm = SVC(kernel='rbf', gamma='scale', class_weight='balanced', C=3)
classifier = make_pipeline(StandardScaler(), svm)
scores = cross_val_score(classifier, data, labels, cv=5)
result = 'RBF Kernel Scores: {}'.format(
    'Accuracy: %0.2f (+/- %0.2f)' % (scores.mean(), scores.std() * 2))
print(result)

svm = SVC(kernel='sigmoid', gamma='scale', C=1)
classifier = make_pipeline(StandardScaler(), svm)
scores = cross_val_score(classifier, data, labels, cv=5)
result = 'Sigmoid Kernel Scores: {}'.format(
    'Accuracy: %0.2f (+/- %0.2f)' % (scores.mean(), scores.std() * 2))
print(result)


track_ids, data, labels, user_names, track_dict, playlist_dict = get_track_data(audio_features=True, genres=True, artists=True, datasets=['spotify_user_data_set'], users=['Donald Duberstein', 'Amel Awadelkarim', 'Micah St Clair'], transform_glove=True)
print(data.shape)
svm = SVC(kernel='rbf', gamma='scale', C=3)
classifier = make_pipeline(StandardScaler(), svm)
scores = cross_val_score(classifier, data, labels, cv=5)
result = 'RBF Kernel Scores: {}'.format(
    'Accuracy: %0.2f (+/- %0.2f)' % (scores.mean(), scores.std() * 2))
print(result)
"""
