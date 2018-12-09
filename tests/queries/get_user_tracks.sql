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
	pl.name as playlist_name
FROM "Playlists" pl
INNER JOIN playlist_track pt ON pl.id = pt."playlistId"
INNER JOIN "Tracks" tr ON pt."trackId" = tr.id
INNER JOIN "AudioFeatures" af ON af."trackId" = tr.id
INNER JOIN artist_track at ON tr.id = at."trackId"
INNER JOIN artist_genre ag ON at."artistId" = ag."artistId"
WHERE pl."ownerId" != 'spotify'
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
	pl.name;
