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
	pl.id as playlist_id
FROM "Playlists" pl
INNER JOIN playlist_track pt ON pl.id = pt."playlistId"
INNER JOIN "Tracks" tr ON pt."trackId" = tr.id
INNER JOIN "AudioFeatures" af ON af."trackId" = tr.id;