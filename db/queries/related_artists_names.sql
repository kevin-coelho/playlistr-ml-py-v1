SELECT
	a1.name AS primary_name,
	a2.name AS secondary_name
FROM related_artist ra
INNER JOIN "Artists" a1 on ra."primaryArtist" = a1.id
INNER JOIN "Artists" a2 on ra."secondaryArtist" = a2.id;