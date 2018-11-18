import psycopg2
import numpy as np

try:
    conn = psycopg2.connect("dbname='playlistr_ml_v1' user='playlistr_ml_v1' host='localhost' password='plt_210'")
    cur = conn.cursor()
    try:
        # open the query file in "read mode" ('r'), read it (read()), then execute that query
        cur.execute(open('get_data_set.sql', 'r').read())
        # get all the rows from the result of executing the query
        rows = cur.fetchall()
        # for every row, split off the first and last columns (trackId and playlistId), then convert to numpy array
        data_arr = np.array([row[1:-1] for row in rows])
        print(data_arr)
        """
        print('\nRows: \n')
        print(rows[0:5])"""

    except Exception as e:
        print(e)

except Exception as e:
    print(e)
