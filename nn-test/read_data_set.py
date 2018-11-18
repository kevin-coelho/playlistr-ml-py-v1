import psycopg2
import numpy as np

try:
    conn = psycopg2.connect("dbname='playlistr_ml_v1' user='playlistr_ml_v1' host='localhost' password='plt_210'")
    cur = conn.cursor()
    try:
        # open the query file in "read mode" ('r'), read it (read()), then execute that query
        cur.execute(open('get_track_features.sql', 'r').read())
        features = cur.fetchall()
        features_array = np.array([row for row in features])

        cur.execute(open('get_track_genres.sql', 'r').read())
        genres = cur.fetchall()
        genre_array = np.array([row for row in genres])

        cur.execute(open('get_playlist_tracks.sql', 'r').read())
        playlists = cur.fetchall()
        playlist_array = np.array([row for row in playlists])
        # for every row, split off the first and last columns (trackId and playlistId), then convert to numpy array
        print("Features query: ", features_array)
        print("Genres query: ", genre_array)
        print("Playlists query: ", playlist_array)
        print("Unique playlists: ", set(playlist_array[:][0]))

    except Exception as e:
        print(e)

except Exception as e:
    print(e)
