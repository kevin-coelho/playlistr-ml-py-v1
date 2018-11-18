import numpy as np
import psycopg2
import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # an affine operation: y = Wx + b
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
    unique_tracks = set(features_array[:][0])
    track_to_int = {list(unique_tracks)[i]: i for i in range(len(list(unique_tracks)))}

    cur.execute(open('get_track_genres.sql', 'r').read())
    genres = cur.fetchall()
    genre_array = np.array([row for row in genres])
    track_genres = {track: [] for track in unique_tracks}
    for row in genre_array:
        track_genres[row[0]].append(row[1])

    cur.execute(open('get_playlist_tracks.sql', 'r').read())
    playlists = cur.fetchall()
    playlist_array = np.array([row for row in playlists])
    unique_playlists = set(playlist_array[:][0])
    playlist_to_int = {list(unique_playlists)[i]: i for i in range(len(list(unique_playlists)))}

    print("Features query: ", features_array)
    print("Genres query: ", genre_array)
    print("Playlists query: ", playlist_array)
    # print("Unique playlists: ", set(playlist_array[:][0]))

    # for epoch in range(2):  # loop over the dataset multiple times
    #     running_loss = 0.0
    #     # for i, data in enumerate(trainloader, 0):
    #     for sample in features_array
    #         # get the inputs
    #         # inputs, labels = data
    #
    #         # zero the parameter gradients
    #         optimizer.zero_grad()
    #
    #         # forward + backward + optimize
    #         output = net(sample[1:])
    #         loss = criterion(output, label)
    #         loss.backward()
    #         optimizer.step()
    #
    #         # print statistics
    #         running_loss += loss.item()
    #         if i % 2000 == 1999:    # print every 2000 mini-batches
    #             print('[%d, %5d] loss: %.3f' %
    #                   (epoch + 1, i + 1, running_loss / 2000))
    #             running_loss = 0.0
    #
    # print('Finished Training')


if __name__ == '__main__':
    main()
