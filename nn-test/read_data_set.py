import psycopg2
import numpy as np

try:
    conn = psycopg2.connect(
        "dbname='playlistr_ml_v1' user='playlistr_ml_v1' host='localhost' password='plt_210'")
    cur = conn.cursor()

except Exception as e:
    print(e)


def get_tracks():
    try:
        # open the query file in "read mode" ('r'), read it (read()), then
        # execute that query
        cur.execute(open('./queries/get_tracks.sql', 'r').read())
        # get all the rows from the result of executing the query
        rows = cur.fetchall()
        # for every row, split off the first and last columns (trackId and playlistId), then convert to numpy array
        # print(rows)
        return rows

    except Exception as e:
        print(e)


def get_genres():
    try:
        # open the query file in "read mode" ('r'), read it (read()), then
        # execute that query
        cur.execute('SELECT name FROM "Genres";')
        # get all the rows from the result of executing the query
        rows = cur.fetchall()
        # print(rows)
        return rows

    except Exception as e:
        print(e)


def get_toy_set():
    """
        Get toy set as numpy arrays (m samples X k features / sample), return dict:
        {
            'data': all data (not split into train / test),
            'labels': labels for every data point
        }
    """
    GENRE_IDX = 11
    TRACK_ID_IDX = 0

    # GET AND FORMAT DATA
    tracks = np.array(get_tracks())

    # get labels
    playlists = tracks[..., -1].flatten()
    playlist_dict = {playlist: idx for idx, playlist in enumerate(set(playlists))}
    labels = np.array([playlist_dict[playlist] for playlist in playlists])

    # init genre dict, convert genres to 1-hot vectors
    genre_dict = {genre[0]: idx for idx, genre in enumerate(get_genres())}
    genres = tracks[..., GENRE_IDX].flatten()
    one_hot_genres = [[0 for genre in range(
        len(genre_dict))] for idx in range(len(tracks))]
    for genre_list, one_hot_arr in zip(genres, one_hot_genres):
        for genre_name in genre_list:
            if genre_name:
                one_hot_arr[genre_dict[genre_name]] += 1
    tracks = np.delete(
        tracks, [TRACK_ID_IDX, GENRE_IDX, tracks.shape[1] - 1], axis=1)
    data_arr = np.append(tracks, one_hot_genres, axis=1)

    # remove nulls
    for sample in data_arr:
        for idx, elem in enumerate(sample):
            if elem is None:
                sample[idx] = 0

    return {
        'data_arr': data_arr,
        'labels': labels,
    }


def get_toy_set_genre_only():
    """
        Get toy set as numpy arrays (m samples X k genres / sample), return dict:
        {
            'data': all data (not split into train / test),
            'labels': labels for every data point
        }
    """
    GENRE_IDX = 11

    # GET AND FORMAT DATA
    tracks = np.array(get_tracks())

    # get labels
    playlists = tracks[..., -1].flatten()
    playlist_dict = {playlist: idx for idx, playlist in enumerate(set(playlists))}
    labels = np.array([playlist_dict[playlist] for playlist in playlists])

    # init genre dict, convert genres to 1-hot vectors
    genre_dict = {genre[0]: idx for idx, genre in enumerate(get_genres())}
    genres = tracks[..., GENRE_IDX].flatten()
    one_hot_genres = [[0 for genre in range(
        len(genre_dict))] for idx in range(len(tracks))]
    for genre_list, one_hot_arr in zip(genres, one_hot_genres):
        for genre_name in genre_list:
            if genre_name:
                one_hot_arr[genre_dict[genre_name]] += 1

    return {
        'data_arr': one_hot_genres,
        'labels': labels,
    }

def get_user_set():
    """
        Get toy set as numpy arrays (m samples X k features / sample), return dict:
        {
            'data': all data (not split into train / test),
            'labels': labels for every data point
        }
    """
    GENRE_IDX = 11
    TRACK_ID_IDX = 0

    # GET AND FORMAT DATA
    tracks = np.array(get_tracks())

    # get labels
    playlists = tracks[..., -1].flatten()
    playlist_dict = {playlist: idx for idx, playlist in enumerate(set(playlists))}
    labels = np.array([playlist_dict[playlist] for playlist in playlists])

    # init genre dict, convert genres to 1-hot vectors
    genre_dict = {genre[0]: idx for idx, genre in enumerate(get_genres())}
    genres = tracks[..., GENRE_IDX].flatten()
    one_hot_genres = [[0 for genre in range(
        len(genre_dict))] for idx in range(len(tracks))]
    for genre_list, one_hot_arr in zip(genres, one_hot_genres):
        for genre_name in genre_list:
            if genre_name:
                one_hot_arr[genre_dict[genre_name]] += 1
    tracks = np.delete(
        tracks, [TRACK_ID_IDX, GENRE_IDX, tracks.shape[1] - 1], axis=1)
    data_arr = np.append(tracks, one_hot_genres, axis=1)

    # remove nulls
    for sample in data_arr:
        for idx, elem in enumerate(sample):
            if elem is None:
                sample[idx] = 0

    return {
        'data_arr': data_arr,
        'labels': labels,
    }
