# DEPENDENCIES
import sys
import os
import numpy as np

# MODULE DEPS
CWD = os.path.dirname(os.path.realpath(__file__))
DB_FOLDER = os.path.realpath(os.path.join(CWD, '../db'))
GLOVE_MODEL_FOLDER = os.path.realpath(os.path.join(CWD, '../related-artists'))
sys.path.append(DB_FOLDER)
sys.path.append(GLOVE_MODEL_FOLDER)
import generate_related_artists_glove
import generate_related_genres_glove
from read_data_set import get_tracks


def get_related_glove_models():
    return (generate_related_artists_glove.get_glove_model(), generate_related_genres_glove.get_glove_model())


def get_track_data(audio_features, genres, artists, users=None, playlists=None, datasets=None, transform_glove=False):
    query_res = np.array(get_tracks(audio_features, genres, artists, users, playlists, datasets))

    # get appropriate indices
    AUDIO_INDICES = range(1, 10)
    if genres:
        GENRE_IDX = 11 if audio_features else 1
    if artists:
        if audio_features and genres:
            ARTIST_IDX = 12
        elif audio_features:
            ARTIST_IDX = 11
        else:
            ARTIST_IDX = 1

    if audio_features:
        audio_res = tracks[..., [i for i in AUDIO_INDICES]]  # rows X audio features
    if genres:
        genre_res = tracks[..., GENRE_IDX]  # rows X 1 (genre array)
    if artists:
        artist_res = tracks[..., ARTIST_IDX]  # rows X 1 (artist array)
    if users:
        user_res = tracks[..., -2]  # rows X 1 (user display name)

    # get tracks
    track_res = tracks[..., 1]  # rows X 1 (track name)
    # get labels
    playlist_res = tracks[..., -1]  # rows X 1 (playlist name)

    # load glove models
    artist_glove, genres_glove = get_related_glove_models()
    artist_dict, artist_vectors = artist_glove.dictionary, artist_glove.word_vectors
    genre_dict, genre_vectors = genres_glove.dictionary, genres_glove.genre_vectors



    return query_result


tracks = get_track_data(True, True, True, playlists=['sad songs'])
print(tracks[0]['genres'])
