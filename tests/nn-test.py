# NUMPY
import numpy as np

# MODULE DEPS
from read_data_set import *

# NN
import torch
import torch.utils.data as data
import torch.nn as nn
# import torch.nn.functional as F
import torch.optim as optim
from sklearn.model_selection import cross_val_predict
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from skorch import NeuralNet
# from skorch.dataset import Dataset
from torch.autograd import Variable
# from pytorch_data_set import SpotifyDataset

# PLOTTING
import matplotlib.pyplot as plt
import optunity
import optunity.metrics

LR = 0.001
BATCH_SIZE = 2019
EPOCH=1000
#
# # GET FULL DATA
# full_data = get_toy_set()
# full_data_arr = np.asarray(full_data['data_arr']).astype(np.float)
# (numSamples, numFeatures) = full_data_arr.shape
# labels_arr = np.asarray(full_data['labels'])
# numClasses = len(set(labels_arr))

# GET USER DATA
full_data = get_user_set('miz')
full_data_arr = np.asarray(full_data['data_arr']).astype(np.float)
(numSamples, numFeatures) = full_data_arr.shape
labels_arr = np.asarray(full_data['labels'])
numClasses = len(set(labels_arr))

scaler = StandardScaler()
scaled_full_data = scaler.fit_transform(full_data_arr)

# # GET GENRE DATA
# genre_only_data = get_toy_set_genre_only()
# genre_data_arr = np.asarray(genre_only_data['data_arr']).astype(np.float)
# (numGenreSamples, numGenreFeatures) = genre_data_arr.shape

# CONVERT TO PYTORCH DATASET
labels_matrix = np.reshape(labels_arr, (len(labels_arr),1))

enc = OneHotEncoder(handle_unknown='ignore')
y_onehot = enc.fit_transform(labels_matrix).toarray() # for softmax

X = torch.from_numpy(scaled_full_data).float()
y = torch.from_numpy(y_onehot).float()

# @optunity.cross_validated(x=scaled_full_data, y=labels, num_folds=5, num_iter=2)
print(X.shape)
print(y.shape)
train_data = data.TensorDataset(X, y)
loader = data.DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)

X = Variable(X).type(torch.FloatTensor)
y = Variable(y).type(torch.FloatTensor)

# DEFINE NETWORK
class Net(nn.Module):
    def __init__(self, n_features, n_hidden, n_output):
        super(Net, self).__init__()
        # define architecture
        self.hidden = nn.Linear(n_features, n_hidden)
        self.output = nn.Linear(n_hidden, n_output)

    def forward(self, x):
        x = torch.relu(self.hidden(x))
        x = torch.softmax(self.output(x), dim=0)
        return x

if __name__ == '__main__':
    # different nets
    net = Net(numFeatures, 5, numClasses)
    opt = torch.optim.Adam(net.parameters(), lr=LR, betas=(0.9, 0.99))
    loss_func = torch.nn.MSELoss()
    losses_his = []   # record loss

    # training
    for epoch in range(EPOCH):
        running_loss = 0.0
        for step, (b_x, b_y) in enumerate(loader): # for each training step
            output = net(b_x)              # get output for every net
            loss = loss_func(output, b_y)  # compute loss for every net
            opt.zero_grad()                # clear gradients for next train
            loss.backward()                # backpropagation, compute gradients
            opt.step()                     # apply gradients
            # running_loss += loss.item()
            # if step % 20 == 19:
            #     print("[%d, %d]: %.3f" % (epoch + 1, step + 1, running_loss/20))
            #     running_loss = 0.0
        losses_his.append(loss.data.numpy())     # loss recoder
        if epoch % 50 == 0:
            print("[%d]: %.3f" % (epoch, loss.item()))
        # #Accuracy
        # output = (output > 0.5).float()
        # correct = (output == labels).float().sum()
        # print("Epoch {}/{}, Loss: {:.3f}, Accuracy: {:.3f}".format(epoch+1, EPOCH, loss.data[0], correct/output.shape[0]))

    plt.plot(losses_his)
    plt.legend(loc='best')
    plt.xlabel('Steps')
    plt.ylabel('Loss')
    plt.savefig('nntrial-miz.png')
    plt.show()






# # INSTANTIATE PROBLEM
# model = NeuralNet(
#         module = Net,
#         module__numFeatures = numFeatures,
#         module__numClasses = numClasses,
#         criterion = nn.MSELoss,
#         optimizer = optim.SGD,
#         optimizer__lr = 0.001,
#         optimizer__momentum = 0.9,
#         max_epochs = 20)
#
# # train_data.CVSplit(5)
# model.fit(train_data)
# model.save_params(f_params='train_params.pkl')
# y_pred = cross_val_predict(model, X, y, cv=5)
#










# model = Net(numFeatures, 10, numClasses)
#
# # DEFINE LOSS AND MINIMIZATION METHOD
# criterion = nn.MSELoss()
# # criterion = nn.NLLLoss()
# optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
#
# for epoch in range(100):  # loop over the dataset multiple times
#
#     running_loss = 0.0
#     for i, sample in enumerate(trainloader, 0):
#         # get the inputs & targets
#         inputs, labels = sample
#         # print(inputs, labels)
#
#         # zero the parameter gradients
#         optimizer.zero_grad()
#
#         # forward + backward + optimize
#         outputs = model(inputs)
#         loss = criterion(outputs, labels)
#         loss.backward()
#         optimizer.step()
#
#         # print statistics
#         running_loss += loss.item()
#         if i % 500 == 499:    # print every 50 mini-batches
#             print('[%d, %5d] Batch loss: %.3f' %
#                   (epoch + 1, i + 1, running_loss / 500))
#             running_loss = 0.0
#     # print statistics
#     # print('#### [%d] EPOCH LOSS: %.3f' % (epoch + 1, loss.item()))
# print('Finished Training')











#
# # Define train and test data
# batch_size = 32
# train_loader = None  # Change this to training data iterator
# test_loader = None  # Change this to testing data iterator
#
#
# # Checking GPU availability
# use_gpu = torch.cuda.is_available()
#
# # Defining Model
# class Net(nn.Module):
#     def __init__(self):
#         super(Net, self).__init__()
#         self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
#         self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
#         self.conv2_drop = nn.Dropout2d()
#         self.fc1 = nn.Linear(320, 50)
#         self.fc2 = nn.Linear(50, 10)
#
#     def forward(self, x):
#         x = F.relu(F.max_pool2d(self.conv1(x), 2))
#         x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
#         x = x.view(-1, 320)
#         x = F.relu(self.fc1(x))
#         x = F.dropout(x, training=self.training)
#         x = self.fc2(x)
#         return F.log_softmax(x, dim=1)
#
# model = Net()
#
# if use_gpu:
#     model = model.cuda()
#
# optimizer = torch.optim.SGD(model.parameters(), 0.01, 0.9)
#
#
# def train(epoch):
#     model.train()
#     for batch_idx, (data, target) in enumerate(train_loader):
#         if use_gpu:
#             data, target = data.cuda(), target.cuda()
#         data, target = Variable(data), Variable(target)
#
#         optimizer.zero_grad()
#         output = model(data)
#         loss = F.nll_loss(output, target)
#
#         loss.backward()
#         optimizer.step()
#
#         if batch_idx % 50 == 0:
#             print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
#                 epoch, batch_idx * len(data), len(train_loader) * batch_size,
#                 100. * batch_idx / len(train_loader), loss.data[0]))
#
# def test():
#     model.eval()
#     test_loss = 0
#     correct = 0
#     for data, target in test_loader:
#         if use_gpu:
#             data, target = data.cuda(), target.cuda()
#         data, target = Variable(data), Variable(target)
#
#         output = model(data)
#
#         test_loss += F.nll_loss(output, target, size_average=False).data[0] # sum up batch loss
#         pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
#         correct += pred.eq(target.data.view_as(pred)).long().cpu().sum()
#
#     test_loss /= len(test_loader) * batch_size
#     print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
#         test_loss, correct, len(test_loader) * batch_size,
#         100. * correct / (len(test_loader) * batch_size)))
#
#
# for epoch in range(1, 2):
#     train(epoch)
#     test()
