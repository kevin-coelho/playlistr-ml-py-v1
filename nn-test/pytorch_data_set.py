import numpy as np
import psycopg2
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(1, 1)

    def forward(self, x):
        x = self.fc1(x)
        return x

def main():
    # read data from database
    conn = psycopg2.connect("dbname='playlistr_ml_v1' user='playlistr_ml_v1' host='localhost' password='plt_210'")
    cur = conn.cursor()

    cur.execute(open('get_track_features.sql', 'r').read())
    features = cur.fetchall()
    features_array = np.array([row for row in features])
    unique_tracks = set(features_array[:,0])
    track_to_int = {list(unique_tracks)[i]: i for i in range(len(list(unique_tracks)))}

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

    cur.execute(open('get_playlist_tracks.sql', 'r').read())
    playlists = cur.fetchall()
    playlist_array = np.array([row for row in playlists])
    print(playlist_array.shape)
    unique_playlists = set(playlist_array[:][0])
    playlist_to_int = {list(unique_playlists)[i]: i for i in range(len(list(unique_playlists)))}

    track_to_playlist = {}
    for row in playlist_array:
        if row[1] in track_to_playlist.keys():
            track_to_playlist[row[1]].add(row[0])
        else:
            track_to_playlist[row[1]] = set()
            track_to_playlist[row[1]].add(row[0])


    print(track_genres)
    # print(track_to_int)
    # print(playlist_to_int)

    net = Net()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    for epoch in range(2):  # loop over the dataset multiple times
        running_loss = 0.0
        # for i, data in enumerate(trainloader, 0):
        for sample in features_array:
            # get the inputs
            # inputs, labels = data

            # zero the parameter gradients
            print(sample[1:].dtype)
            input = torch.from_numpy(sample[1:])
            # forward + backward + optimize
            output = net(input)
            loss = criterion(output, target)
            loss.backward()
            optimizer.zero_grad()
            optimizer.step()

            # # print statistics
            # running_loss += loss.item()
            # # if i % 2000 == 1999:    # print every 2000 mini-batches
            # print('[%d, %5d] loss: %.3f' %
            #           (epoch + 1, i + 1, running_loss / 2000))
            #     running_loss = 0.0

    print('Finished Training')


if __name__ == '__main__':
    main()
