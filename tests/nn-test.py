# NUMPY
import numpy as np

# MODULE DEPS
from read_data_set import *

# NN
import torch
import torch.utils.data as data
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.utils import shuffle
from torch.autograd import Variable

# PLOTTING
import matplotlib.pyplot as plt

# METRICS
import optunity
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_predict

from math import ceil, floor

# NN PARAMETERS
NAME='jacob'
LR=0.001
EPOCH=500
PRINT_FREQ=10
TWO_LAYERS=False
NLL=True

# GET DATA
full_data = get_user_set(NAME)
full_data_arr = np.asarray(full_data['data_arr']).astype(np.float)
m, n = full_data_arr.shape
print(m,n)

labels_arr = np.asarray(full_data['labels'])
labels_dict = full_data['labels_dict']
C = len(set(labels_arr))

labels_matrix = np.reshape(labels_arr, (len(labels_arr),1))
enc = OneHotEncoder(handle_unknown='ignore')
y_onehot = enc.fit_transform(labels_matrix).toarray() # for softmax

shuffled_data, shuffled_labels = shuffle(full_data_arr, labels_arr) if NLL else shuffle(full_data_arr, y_onehot)
x_train, y_train = (shuffled_data[:floor(0.8*m), :], shuffled_labels[:floor(0.8 * m)]) if NLL else (shuffled_data[:floor(0.8*m), :], shuffled_labels[:floor(0.8 * m), :])
x_test, y_test = (shuffled_data[floor(0.8*m):, :], shuffled_labels[floor(0.8 * m):]) if NLL else (shuffled_data[:floor(0.8*m), :], shuffled_labels[:floor(0.8 * m), :])

scaler = StandardScaler()
scaled_train = scaler.fit_transform(x_train)
scaled_test = scaler.fit_transform(x_test)

# # GET GENRE DATA
# genre_only_data = get_toy_set_genre_only()
# genre_data_arr = np.asarray(genre_only_data['data_arr']).astype(np.float)
# (numGenreSamples, numGenreFeatures) = genre_data_arr.shape

# CONVERT TO PYTORCH DATASET
X_train = torch.from_numpy(scaled_train).float()
Y_train = torch.from_numpy(y_train).long() if NLL else torch.from_numpy(y_train).float()
X_test = torch.from_numpy(scaled_test).float()
Y_test = torch.from_numpy(y_test).long() if NLL else torch.from_numpy(y_test).float()

# @optunity.cross_validated(x=scaled_full_data, y=labels, num_folds=5, num_iter=2)
train_data = data.TensorDataset(X_train, Y_train)
train_loader = data.DataLoader(train_data, batch_size=X_train.shape[0], shuffle=True)
test_data = data.TensorDataset(X_test, Y_test)
test_loader = data.DataLoader(test_data, batch_size=X_test.shape[0], shuffle=True)

# DEFINE NETWORK
class Net(nn.Module):
    def __init__(self, n_features, n_hidden, n_output):
        super(Net, self).__init__()
        # define architecture
        self.hidden = nn.Linear(n_features, n_hidden)
        self.output = nn.Linear(n_hidden, n_output)

    def forward(self, x):
        x = torch.sigmoid(self.hidden(x))
        m = nn.LogSoftmax(dim=0)
        x = m(self.output(x))
        return x

class Net2(nn.Module):
    def __init__(self, n_features, n_hidden1, n_hidden2, n_output):
        super(Net2, self).__init__()
        # define architecture
        self.hidden1 = nn.Linear(n_features, n_hidden1)
        self.hidden2 = nn.Linear(n_hidden1, n_hidden2)
        self.output = nn.Linear(n_hidden2, n_output)

    def forward(self, x):
        x = self.hidden1(x)
        x = torch.sigmoid(self.hidden2(x))
        m = nn.LogSoftmax(dim=0)
        x = m(self.output(x))
        return x


if __name__ == '__main__':
    # different nets
    net = Net2(n, 2*C, C, C) if TWO_LAYERS else Net(n, C, C)
    opt = torch.optim.Adam(net.parameters(), lr=LR, weight_decay=1E-4)
    loss_func = nn.NLLLoss() if NLL else nn.MSELoss()
    train_loss, test_loss = [], []  # record loss
    train_accuracy, test_accuracy = [], []

    def train(epoch):
        net.train()

        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = Variable(data), Variable(target)
            output = net(data)
            loss = loss_func(output, target)
            opt.zero_grad()
            loss.backward()
            opt.step()

        vec_target = target if NLL else target.argmax(dim=0, keepdim=True)
        output = net(data)
        pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
        correct = pred.eq(vec_target.data.view_as(pred)).long().cpu().sum()
        if epoch % PRINT_FREQ == 0:
            train_loss.append(loss.data.numpy())
            train_accuracy.append(100. * correct / (len(train_loader) * X_train.shape[0]))
            print('TRAIN {}\t{:.6f}'.format(epoch, loss.data[0]))

    def test(epoch):
        net.eval()
        running_loss = 0.0
        correct = 0

        for data, target in test_loader:
            data, target = Variable(data), Variable(target)
            vec_target = target if NLL else target.argmax(dim=0, keepdim=True)
            output = net(data)
            running_loss += loss_func(output, target).data[0] # sum up batch loss
            pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(vec_target.data.view_as(pred)).long().cpu().sum()

        running_loss /= len(test_loader) * data.shape[0]
        if epoch % PRINT_FREQ == 0:
            print('  TEST: Avg. loss: {:.4f}, Accuracy: {:.0f}%\n'.format(running_loss, 100. * correct / (len(test_loader) * X_test.shape[0])))
            test_loss.append(running_loss)
            test_accuracy.append(100. * correct / (len(test_loader) * X_test.shape[0]))

        return vec_target, pred


    for epoch in range(EPOCH):
        train(epoch)
        target, output = test(epoch)
    print(confusion_matrix(target, output))
    print(full_data['labels_dict'])

    print('Final Training Accuracy: {}%'.format(float(train_accuracy[-1])))
    print('Final Test Accuracy: {}%'.format(float(test_accuracy[-1])))

    fig, (ax1, ax2) = plt.subplots(2, 1)

    ax1.plot(range(0, EPOCH, PRINT_FREQ), train_loss, 'b', label='train')
    ax1.legend(loc='best')
    ax1.set_ylabel('Loss')

    ax2.plot(range(0, EPOCH, PRINT_FREQ), train_accuracy, 'b', label='train')
    ax2.plot(range(0, EPOCH, PRINT_FREQ), test_accuracy, 'r', label='test')
    ax2.legend(loc='best')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')

    fig.savefig('results/{}4reg2layers500epochC.png'.format(NAME))
    fig.show()
