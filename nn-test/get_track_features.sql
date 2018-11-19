SELECT
	af."trackId" as track_id,
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
	pt."playlistId" as playlist_id
FROM "AudioFeatures" af
INNER JOIN playlist_track pt ON pt."trackId" = af."trackId";
