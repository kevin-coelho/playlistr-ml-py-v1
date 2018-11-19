import models
import numpy as np
import psycopg2
import torch
import torch.nn as nn
import torch.optim as optim
# import torch.nn.functional as F

def main():
    # read data from database
    conn = psycopg2.connect("dbname='playlistr_ml_v1' user='playlistr_ml_v1' host='localhost' password='plt_210'")
    cur = conn.cursor()

    cur.execute(open('get_track_features.sql', 'r').read())
    features = cur.fetchall()
    features_array = np.array([row for row in features])
    tracks = features_array[:, 0]
    X = features_array[:, 1:-1].astype(np.float)
    Y = features_array[:, -1]
    print(tracks, X, Y)
    unique_tracks = set(tracks)
    unique_playlists = set(Y)
    playlist_to_int = {list(unique_playlists)[i]: i for i in range(len(list(unique_playlists)))}
    # track_to_int = {list(unique_tracks)[i]: i for i in range(len(list(unique_tracks)))}

    cur.execute(open('get_track_genres.sql', 'r').read())
    genres = cur.fetchall()
    genre_array = np.array([row for row in genres])
    track_genres = {}
    for row in genre_array:
        if row[0] in track_genres.keys():
            track_genres[row[0]].add(row[1])
        else:
            track_genres[row[0]] = set()
            track_genres[row[0]].add(row[1])

    # cur.execute(open('get_playlist_tracks.sql', 'r').read())
    # playlists = cur.fetchall()
    # playlist_array = np.array([row for row in playlists])


    track_to_playlist = {row[1]: row[0] for row in playlist_array}
    # for row in playlist_array:
        # if row[1] in track_to_playlist.keys():
        #     track_to_playlist[row[1]].add(row[0])
        # else:
        #     track_to_playlist[row[1]] = set()
        #     track_to_playlist[row[1]].add(row[0])

    net = models.Net()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    for epoch in range(2):  # loop over the dataset multiple times
        running_loss = 0.0
        # for i, data in enumerate(trainloader, 0):
        m,n = X.shape
        for i in range(m):
            # get the inputs
            # inputs, labels = data

            # zero the parameter gradients
            input = torch.from_numpy(X[i,:])
            # forward + backward + optimize
            output = net(input.float())
            # loss = criterion(output, target)
            # loss.backward()
            # optimizer.zero_grad()
            # optimizer.step()

            # print statistics
            # running_loss += loss.item()
            # # if i % 2000 == 1999:    # print every 2000 mini-batches
            # print('[%d, %5d] loss: %.3f' %
            #           (epoch + 1, i + 1, running_loss / 2000))
            #     running_loss = 0.0

    print('Finished Training')


if __name__ == '__main__':
    main()
