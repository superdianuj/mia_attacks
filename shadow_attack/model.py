import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, SubsetRandomSampler, TensorDataset, ConcatDataset
import numpy as np
import os
from sklearn.metrics import classification_report
import sklearn
import matplotlib.pyplot as plt
import seaborn as sns
import torch.nn.functional as F


# I am associating a set of shadow models for each target sample, so for N-samples, I am create N-set of K shadow models
class CNN(nn.Module):
    def __init__(self, channel, num_classes):
        super(CNN, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(channel, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        # Pooling layer
        self.pool = nn.MaxPool2d(2, 2)
        
        # Dropout layer
        self.dropout = nn.Dropout(0.5)
        
        # Calculating the size of the input for the first fully connected layer
        fc1_input_size = 1024
        
        # Fully connected layers
        self.fc1 = nn.Linear(fc1_input_size, 1024)
        self.fc2 = nn.Linear(1024, num_classes)
        
    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.pool(F.relu(self.bn4(self.conv4(x))))
        
        # Flattening the output
        x = x.view(x.size(0), -1)
        
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

    def feature(self,x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.pool(F.relu(self.bn4(self.conv4(x))))
        
        # Flattening the output
        x = x.view(x.size(0), -1)

        return x
    

class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(MLP, self).__init__()
        self.fc1 = nn.Sequential(nn.Linear(input_size, hidden_size))
        self.hidden1 = nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.ReLU())
        self.hidden2 = nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.Sigmoid())
        self.fc2 = nn.Sequential(nn.Linear(hidden_size, num_classes))
    
    def forward(self, x):
        out = self.fc1(x)
        out = self.hidden1(out)
        out = self.hidden2(out)
        out = self.fc2(out)
        return out