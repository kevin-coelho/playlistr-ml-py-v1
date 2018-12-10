# DEPENDENCIES
import sys
import os
import numpy as np
import pdb

# MODULE DEPS
CWD = os.path.dirname(os.path.realpath(__file__))
DB_FOLDER = os.path.realpath(os.path.join(CWD, '../db'))
GLOVE_MODEL_FOLDER = os.path.realpath(os.path.join(CWD, '../related-artists'))
sys.path.insert(0, DB_FOLDER)
sys.path.insert(0, GLOVE_MODEL_FOLDER)
import generate_related_artists_glove
import generate_related_genres_glove
from read_data_set import get_tracks, get_audio_features


def get_related_glove_models():
    return (generate_related_artists_glove.get_glove_model(), generate_related_genres_glove.get_glove_model())


def compute_vectors(input_arr, vector_dict, vectors):
    avg_vec = np.mean(vectors, axis=0)
    transformed = np.ndarray((len(input_arr), vectors.shape[1]))
    for idx, arr in enumerate(input_arr):
        arr = [elem for elem in arr if elem is not None]
        if len(arr) < 1:
            vec = avg_vec
        else:
            vec = np.zeros(vectors.shape[1])
        for elem in arr:
            if elem and elem in vector_dict:
                vec += vectors[vector_dict[elem]]
        transformed[idx] = vec
    return transformed


def generate_one_hot(input_arr, input_dict):
    output = np.zeros((len(input_arr), len(input_dict)))
    for idx, arr in enumerate(input_arr):
        for elem in arr:
            if elem and elem in input_dict:
                output[idx][input_dict[elem]] += 1
    return output


def extract_audio_features(res, avg_features):
    audio_res = np.ndarray((len(res), len(avg_features)))
    for sample_idx, row in enumerate(res):
        for idx in range(1, 11):
            if row[idx] is not None:
                audio_res[sample_idx][idx - 1] = row[idx]
            else:
                audio_res[sample_idx][idx - 1] = avg_features[idx - 1]
    return audio_res


def extract_genres(res, idx):
    genre_res = []
    for row_idx, row in enumerate(res):
        genre_res.append([])
        for genre in row[idx]:
            if genre is not None:
                genre_res[row_idx] += [genre]
    return genre_res


def extract_artists(res, idx):
    artist_res = []
    for row_idx, row in enumerate(res):
        artist_res.append([])
        for artist in row[idx]:
            if artist is not None:
                artist_res[row_idx] += [artist]
    return artist_res


def extract_user_names(res):
    user_names = []
    for idx, row in enumerate(res):
        user_names.append(row[-2])
    return user_names


def get_track_data(audio_features=True, genres=True, artists=False, users=None, playlists=None, datasets=None, transform_glove=False):
    query_res = get_tracks(audio_features, genres, artists, users, playlists, datasets)

    if audio_features:
        avg_audio_features = np.mean(np.array(get_audio_features()), axis=0)
        audio_res = extract_audio_features(query_res, avg_audio_features)

    if genres:
        GENRE_IDX = 11 if audio_features else 1
        genre_res = extract_genres(query_res, GENRE_IDX)

    if artists:
        if audio_features and genres:
            ARTIST_IDX = 12
        elif audio_features:
            ARTIST_IDX = 11
        else:
            ARTIST_IDX = 1
        artist_res = extract_artists(query_res, ARTIST_IDX)
    user_names = [row[-2] for row in query_res]
    track_ids = [row[0] for row in query_res]
    playlist_ids = [row[-1] for row in query_res]

    # load glove models
    artist_glove, genres_glove = get_related_glove_models()
    artist_dict, artist_vectors = artist_glove.dictionary, artist_glove.word_vectors
    genre_dict, genre_vectors = genres_glove.dictionary, genres_glove.word_vectors

    if transform_glove:
        if genres:
            genres_transformed = compute_vectors(genre_res, genre_dict, genre_vectors)
        if artists:
            artists_transformed = compute_vectors(artist_res, artist_dict, artist_vectors)
    else:
        if genres:
            genres_transformed = generate_one_hot(genre_res, genre_dict)

    data = []
    for idx, track in enumerate(track_ids):
        data.append([])
        if audio_features:
            features = audio_res[idx]
            data[idx].extend(features)
        if genres:
            data[idx].extend(genres_transformed[idx])
        if artists and transform_glove:
            data[idx].extend(artists_transformed[idx])
    data = np.array(data)
    return (track_ids, data, playlist_ids, user_names)
