import psycopg2
import psycopg2.extras
import numpy as np
import os

CWD = os.path.dirname(os.path.realpath(__file__))
QUERY_FOLDER = os.path.join(CWD, 'queries')


#     array_agg(ag."genre") as genres,

try:
    conn = psycopg2.connect(
        "dbname='playlistr_ml_v1' user='playlistr_ml_v1' host='localhost' password='plt_210'")
    cur = conn.cursor()

except Exception as e:
    print(e)
    raise e


def get_playlist_dict(playlists=None, datasets=None):
    SELECT = '''SELECT pl.id, pl.name FROM dataset_playlist dp INNER JOIN "Playlists" pl on dp."playlistId" = pl."id"'''

    first_where = False
    WHERE = ''
    if datasets:
        DATASET_WHERE = '''dp."datasetName" IN ({})'''.format(','.join(map(lambda s: '\'' + str(s) + '\'', datasets)))
        WHERE += 'WHERE ' + DATASET_WHERE
        first_where = True
    if playlists:
        PLAYLIST_WHERE = '''pl."id" IN ({})'''.format(','.join(map(lambda s: '\'' + str(s) + '\'', playlists)))
        WHERE += '\n' + ('AND ' if first_where else 'WHERE ') + PLAYLIST_WHERE
        first_where = True

    QUERY = '{}\n{};'.format(SELECT, WHERE)
    cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
    cur.execute(QUERY)
    rows = cur.fetchall()
    playlist_dict = {row[0]: row[1] for row in rows}
    return playlist_dict


def get_track_dict(tracks):
    SELECT = '''SELECT tr.id, tr."name" FROM "Tracks" tr'''
    WHERE = '''WHERE tr."id" IN ({})'''.format(','.join(map(lambda s: '\'' + str(s) + '\'', tracks)))
    QUERY = '{}\n{};'.format(SELECT, WHERE)
    cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
    cur.execute(QUERY)
    rows = cur.fetchall()
    track_dict = {row[0]: row[1] for row in rows}
    return track_dict


def compose_select_tracks(audio_features, genres, artists, users=None, playlists=None, datasets=None):
    if not playlists and not datasets:
        raise Exception('One of playlists array or datasets array must be provided')
    if users and len(users) < 1:
        raise Exception('Empty users array provided')
    if playlists and len(playlists) < 1:
        raise Exception('Empty playlists array provided')
    if datasets and len(datasets) < 1:
        raise Exception('Empty datasets array provided')

    SELECT = 'SELECT \n    tr.id AS track_id'
    if audio_features:
        AUDIO_FEATURES = '''
,\n    af.danceability,
    af.energy,
    af.mode,
    af.speechiness,
    af.acousticness,
    af.instrumentalness,
    af.liveness,
    af.valence,
    af.tempo,
    af.time_signature
        '''.strip()
        SELECT += AUDIO_FEATURES
    if genres:
        SELECT += ',\n    array_agg(ag."genre") as genres'
    if artists:
        SELECT += ',\n    array_agg(ar."name") as artists'
    if users and len(users) > 0:
        SELECT += ',\n    us."display_name" as owner_name'
    SELECT += ',\n    pl.id as playlist_id'

    FROM = '''
FROM dataset_playlist dp
INNER JOIN "Playlists" pl on dp."playlistId" = pl."id"
INNER JOIN playlist_track pt ON pl.id = pt."playlistId"
INNER JOIN "Tracks" tr ON pt."trackId" = tr.id
{audio_features}
{artist_track}
{genres}
{artists}
{owners}
    '''.format(
        artist_track=('INNER JOIN artist_track at ON tr.id = at."trackId"' if genres or artists else ''),
        audio_features=('INNER JOIN "AudioFeatures" af ON af."trackId" = tr.id' if audio_features else ''),
        genres=('INNER JOIN artist_genre ag ON at."artistId" = ag."artistId"' if genres else ''),
        artists=('INNER JOIN "Artists" ar ON at."artistId" = ar.id' if artists else ''),
        owners=('INNER JOIN "Users" us on pl."ownerId" = us.id' if users and len(users) > 0 else '')
    ).strip()

    first_where = False
    WHERE = ''
    if datasets:
        DATASET_WHERE = '''dp."datasetName" IN ({})'''.format(','.join(map(lambda s: '\'' + str(s) + '\'', datasets)))
        WHERE += 'WHERE ' + DATASET_WHERE
        first_where = True
    if playlists:
        PLAYLIST_WHERE = '''pl."name" IN ({})'''.format(','.join(map(lambda s: '\'' + str(s) + '\'', playlists)))
        WHERE += '\n' + ('AND ' if first_where else 'WHERE ') + PLAYLIST_WHERE
        first_where = True
    if users:
        USER_WHERE = '''us."display_name" IN ({})'''.format(','.join(map(lambda s: '\'' + str(s) + '\'', users)))
        WHERE += '\n' + 'AND ' + USER_WHERE

    GROUP_BY = 'GROUP BY\n    tr.id,\n    pl.id'
    if audio_features:
        GROUP_BY += AUDIO_FEATURES
    if users:
        GROUP_BY += ',\n    us."display_name"'

    ORDER_BY = 'ORDER BY tr.id DESC'
    return '{}\n{}\n{}\n{}\n{};'.format(SELECT, FROM, WHERE, GROUP_BY, ORDER_BY)


def get_tracks(audio_features, genres, artists, users=None, playlists=None, datasets=None):
    try:
        cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
        cur.execute(compose_select_tracks(audio_features, genres, artists, users, playlists, datasets))
        rows = cur.fetchall()
        return rows
    except Exception as e:
        print(e)
        return None


def get_genres():
    try:
        cur.execute('SELECT name FROM "Genres";')
        rows = cur.fetchall()
        return rows

    except Exception as e:
        print(e)
        return None


def get_related_genres():
    try:
        cur.execute('SELECT "artistId", array_agg(genre) FROM artist_genre GROUP BY "artistId";')
        rows = cur.fetchall()
        return rows
    except Exception as e:
        print(e)
        return None


def get_related_artists():
    try:
        QUERY_FILE = os.path.join(QUERY_FOLDER, 'related_artists_names.sql')
        cur.execute(open(QUERY_FILE, 'r').read())
        rows = cur.fetchall()
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
