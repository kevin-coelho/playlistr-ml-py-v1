import lyricsgenius
genius = lyricsgenius.Genius('ym7RZPLBCn_rVlY2F14a-wczJ0TF9P4ostyu8AbMOoatvJjwx6Elag8hACCSdxcY')

# CONFIG
genius.verbose = False # Turn off status messages
genius.remove_section_headers = True # Remove section headers (e.g. [Chorus]) from lyrics when searching
genius.skip_non_songs = False # Include hits thought to be non-songs (e.g. track lists)
# genius.excluded_terms = ["(Remix)", "(Live)"] # Exclude songs with these words in their title

song = genius.search_song('Tease Me', 'Lianne La Havas')
print(song.lyrics)
