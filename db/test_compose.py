from read_data_set import compose_select_tracks
"""
print(compose_select_tracks(True, True, True, users=['Kevin Coelho'], playlists=['sad songs'], datasets=['spotify_user_data_set']))
print(compose_select_tracks(False, True, False, users=['Kevin Coelho'], playlists=['sad songs']))"""
print(compose_select_tracks(False, False, True, users=['Kevin Coelho'], playlists=['sad songs', 'indie soundcloud']))
print(compose_select_tracks(False, False, True, datasets=['spotify_user_data_set']))

# def compose_select_tracks(audio_features, genres, artists, users=None, playlists=None, datasets=None):
