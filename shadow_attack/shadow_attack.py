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


class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, data, targets):
        self.data = data
        self.targets = targets

    def __getitem__(self, index):
        x = self.data[index]
        y = self.targets[index]
        return x, y

    def __len__(self):
        return len(self.data)



# Load CIFAR-10 dataset
def get_data_loaders(batch_size, train_r=0.01, test_r=0.9, num_shadow_models=5):
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    
    train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    
    train_size = len(train_set)
    indices = list(range(train_size))
    np.random.shuffle(indices)
    shadow_split = int(np.floor(train_r * train_size))

    test_size = len(test_set)
    test_indices = list(range(test_size))
    np.random.shuffle(test_indices)
    shadow_split_test = int(np.floor(test_r * test_size))

    train_images = [train_set[i][0] for i in indices]
    train_labels = [train_set[i][1] for i in indices]
    test_images = [test_set[i][0] for i in test_indices]
    test_labels = [test_set[i][1] for i in test_indices]

    shadow_datasets = []
    for i in range(num_shadow_models):
        shadow_in=CustomDataset(train_images[:shadow_split],train_labels[:shadow_split])
        shadow_datasets.append(shadow_in)

    shadow_out=CustomDataset(test_images[:shadow_split_test ],test_labels[:shadow_split_test])

    shadow_loaders = [DataLoader(dataset, batch_size=batch_size, shuffle=True) for dataset in shadow_datasets]
    target_loader = DataLoader(train_set, batch_size=batch_size)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
    shadow_loader_test = DataLoader(shadow_out, batch_size=batch_size, shuffle=True)
    return target_loader, shadow_loaders, test_loader, shadow_loader_test

# Train a model
def train_model(model, train_loader, criterion, optimizer, num_epochs, device):
    model.train()
    for epoch in range(num_epochs):
        run_loss = 0.0
        for images, labels in train_loader:
            images = images.view(images.size(0), -1).to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            run_loss += loss.item()
        
        if epoch % 10 == 0 or epoch == num_epochs - 1 or epoch == 0:
            print(f'Epoch {epoch+1}/{num_epochs}, Loss: {run_loss/len(train_loader):.4f}')

# Prepare attack data
def prepare_attack_data(model, data_loader, device, is_member):
    model.eval()
    attack_X = []
    attack_Y = []
    with torch.no_grad():
        for images, labels in data_loader:
            images = images.view(images.size(0), -1).to(device)
            outputs = model(images)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            attack_X.append(probs.cpu())
            attack_Y.append(torch.ones(probs.size(0)) if is_member else torch.zeros(probs.size(0)))
    attack_X = torch.cat(attack_X)
    attack_Y = torch.cat(attack_Y).long()
    return attack_X, attack_Y

# Train the attack model
def train_attack_model(model, train_loader, criterion, optimizer, num_epochs, device):
    model.train()
    for epoch in range(num_epochs):
        run_loss = 0.0
        for features, labels in train_loader:
            features = features.to(device)
            labels = labels.to(device)
            
            outputs = model(features)
            loss = criterion(outputs, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            run_loss += loss.item()
        if epoch % 10 == 0 or epoch == num_epochs - 1 or epoch == 0:
            print(f'Epoch {epoch+1}/{num_epochs}, Loss: {run_loss/len(train_loader):.4f}')

# Evaluate the attack model
def evaluate_attack_model(model, data_loader, device):
    model.eval()
    true_labels = []
    pred_labels = []
    with torch.no_grad():
        for features, labels in data_loader:
            features = features.to(device)
            labels = labels.to(device)
            
            outputs = model(features)
            _, preds = torch.max(outputs, 1)
            true_labels.append(labels.cpu())
            pred_labels.append(preds.cpu())
    
    true_labels = torch.cat(true_labels)
    pred_labels = torch.cat(pred_labels)
    print(classification_report(true_labels, pred_labels, target_names=['Non-Member', 'Member']))
    auc = sklearn.metrics.roc_auc_score(true_labels, pred_labels)
    print(f'AUC: {auc:.4f}')
    return auc

def create_attack_loader(shadow_models, target_model, shadow_loaders, shadow_loader_test, attack_batch_size):
    attack_X = []
    attack_Y = []
    
    for shadow_model, shadow_loader in zip(shadow_models, shadow_loaders):
        shadow_X, shadow_Y = prepare_attack_data(shadow_model, shadow_loader, device, is_member=True)
        attack_X.append(shadow_X)
        attack_Y.append(shadow_Y)
    
    target_X, target_Y = prepare_attack_data(target_model, shadow_loader_test, device, is_member=False)
    attack_X.append(target_X)
    attack_Y.append(target_Y)

    attack_X = torch.cat(attack_X)
    attack_Y = torch.cat(attack_Y)

    attack_dataset = TensorDataset(attack_X, attack_Y)
    attack_loader = DataLoader(attack_dataset, batch_size=attack_batch_size, shuffle=True)

    return attack_loader

def get_model_accuracy(model, data_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in data_loader:
            images = images.view(images.size(0), -1).to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()
    return correct / total * 100