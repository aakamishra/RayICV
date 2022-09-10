from torchvision import datasets, transforms
from torchvision import transforms
from api import RayCrossValidation
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import torch
import torch.nn as nn


# define MNIST baseline model
class MNISTNet(nn.Module):
    def __init__(self, config):
        d1, d2, = 0.25, 0.5
        if 'd1' in config:
            d1 = config['d1']
        if 'd2' in config:
            d2 = config['d2']
        print('model params: ', d1, d2)
        super(MNISTNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(d1)
        self.dropout2 = nn.Dropout(d2)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


# define transform for dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# load mnist data
train_data = datasets.MNIST(root='data', train=True, transform=transform)
test_data = datasets.MNIST(root='data', train=False, transform=transform)

print('Confirm Data-LoadOp Was Successful')
print('==================================')
print(train_data)
print(train_data.data.size())
print(test_data)
print(train_data.targets.size())

# run distributed cross validation
print('About to run distributed RayCrossValidation')
optimizer = optim.Adadelta
parameters = {'d1' : [0.01, 0.1, 0.25, 0.3], 'd2' : [0.1, 0.3, 0.5]}
best_config = RayCrossValidation(MNISTNet, train_data, parameters, 5,
                                 optimizer=optimizer, epochs=1)
print('Best Config', best_config)
