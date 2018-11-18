SELECT
	at."trackId" as track_id,
	ag.genre
FROM artist_track at
INNER JOIN artist_genre ag ON ag."artistId" = at."artistId"
