SELECT
	tr.id as track_id,
	af.danceability,
	af.energy,
	af.mode,
	af.speechiness,
	af.acousticness,
	af.instrumentalness,
	af.liveness,
	af.valence,
	af.tempo,
	af.time_signature,
	array_agg(ag."genre") as genres,
	pl.id as playlist_id
FROM dataset_playlist dp
INNER JOIN "Playlists" pl on dp."playlistId" = pl."id"
INNER JOIN playlist_track pt ON pl.id = pt."playlistId"
INNER JOIN "Tracks" tr ON pt."trackId" = tr.id
INNER JOIN "AudioFeatures" af ON af."trackId" = tr.id
INNER JOIN artist_track at ON tr.id = at."trackId"
INNER JOIN artist_genre ag ON at."artistId" = ag."artistId"
WHERE dp."datasetName" = 'spotify_toy_data_set'
GROUP BY
	tr.id,
	af.danceability,
	af.energy,
	af.mode,
	af.speechiness,
	af.acousticness,
	af.instrumentalness,
	af.liveness,
	af.valence,
	af.tempo,
	af.time_signature,
	pl.id;
