# NUMPY
import numpy as np

# MODULE DEPS
from read_data_set import get_toy_set, get_toy_set_genre_only

# NN
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from skorch import NeuralNet
from torch.autograd import Variable
from pytorch_data_set import SpotifyDataset


# GET FULL DATA
full_data = get_toy_set()
full_data_arr = np.asarray(full_data['data_arr']).astype(np.float)
(numSamples, numFeatures) = full_data_arr.shape
labels_arr = np.asarray(full_data['labels'])
numClasses = len(set(labels_arr))

# # GET GENRE DATA
# genre_only_data = get_toy_set_genre_only()
# genre_data_arr = np.asarray(genre_only_data['data_arr']).astype(np.float)
# (numGenreSamples, numGenreFeatures) = genre_data_arr.shape

# CONVERT TO PYTORCH DATASET
labels_matrix = np.reshape(labels_arr, (len(labels_arr),1))
X = torch.from_numpy(full_data_arr).float()
y = torch.from_numpy(labels_matrix).long()

train_data = torch.utils.data.TensorDataset(X, y)
trainloader = torch.utils.data.DataLoader(train_data, batch_size=4, shuffle=True)

X = Variable(X).type(torch.FloatTensor)
y = Variable(y).type(torch.LongTensor)

# DEFINE NETWORK
class Net(nn.Module):
    def __init__(self, numFeatures, numClasses):
        super(Net, self).__init__()
        # define architecture
        self.hidden_layer = nn.Linear(numFeatures, numClasses)
        self.output = nn.Linear(numClasses, 1)

    def forward(self, x):
        x = F.sigmoid(self.hidden_layer(x))
        x = F.sigmoid(self.output(x))
        return x








# INSTANTIATE PROBLEM
model = NeuralNet(
        module = Net,
        module__numFeatures = numFeatures,
        module__numClasses = numClasses,
        criterion = nn.NLLLoss,
        optimizer = optim.SGD,
        optimizer__lr = 0.001,
        optimizer__momentum = 0.9)

model.fit_loop(X, y, epochs = 20)
y_pred = model.predict(X_valid)












# model = Net(numFeatures, numClasses)
#
# # DEFINE LOSS AND MINIMIZATION METHOD
# # criterion = nn.MSELoss()
# criterion = nn.NLLLoss()
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
#         if i % 50 == 49:    # print every 50 mini-batches
#             print('[%d, %5d] Batch loss: %.3f' %
#                   (epoch + 1, i + 1, running_loss / 50))
#             running_loss = 0.0
#     # print statistics
#     # print('#### [%d] EPOCH LOSS: %.3f' % (epoch + 1, loss.item()))
# print('Finished Training')












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
