SELECT
    pt."playlistId" as playlist_id,
	at."trackId" as track_id
FROM artist_track at
INNER JOIN playlist_track pt ON pt."trackId" = at."trackId";
