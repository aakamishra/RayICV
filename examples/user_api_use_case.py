from torchvision import datasets, transforms
from torchvision import transforms
from src.api import RayCrossValidation
import numpy as np
import torch
import torch.nn as nn


# define MNIST baseline model
class MNISTNet(nn.Module):
    def __init__(self):
        super(MNISTNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
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
train_data = datasets.MNIST(
    root = 'data',
    train = True,
    transform = transform,
    download = True,
)

test_data = datasets.MNIST(
    root = 'data',
    train = False,
    transform = transform
)


print("Confirm Data-LoadOp Was Successful")
print("==================================")
print(train_data)
print(train_data.data.size())
print(test_data)
print(train_data.targets.size())

# run distributed cross validation
print("About to run distributed RayCrossValidation")
optimizer = optim.Adam(cnn.parameters(), lr = 0.01)
RayCrossValidation(MNISTNet, train_data, 5, optimizer=optimizer, epochs=10)
