import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import transforms
import numpy as np
import torch.nn.functional as F
from scipy.stats import norm
from sklearn.metrics import roc_curve, auc, accuracy_score
from torch.utils.data import DataLoader, SubsetRandomSampler, TensorDataset, ConcatDataset
import math
import matplotlib.pyplot as plt



def calculate_accuracy(model, dataloader, device='cpu'):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            outputs = model(x)
            _, predicted = torch.max(outputs.data, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()
    return 100 * correct / total


# Function to train a model
def train_model_with_raw_tensors(model, train_data, train_labels, epochs=100, lr=0.01,bs=128*2, device='cuda'):
    dataset= torch.utils.data.TensorDataset(train_data, train_labels)
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=bs, shuffle=True)
    criterion = nn.CrossEntropyLoss()
    model.train()
    optimizer = optim.SGD(model.parameters(), lr=lr)
    for epoch in range(epochs):
        for batch in train_loader:
            img, label = batch
            img, label = img.to(device), label.to(device)
            optimizer.zero_grad()
            outputs = model(img)
            loss = criterion(outputs, label)
            loss.backward()
            optimizer.step()
    return model


def train_model_with_loader(model, train_loader, training_epochs=100, lr=0.01, device='cuda'):
    criterion = nn.CrossEntropyLoss()
    optimizer=torch.optim.Adam(model.parameters(), lr=lr)
    for epochy in range(training_epochs):
        for batch in train_loader:
            data, target = batch[0].to(device), batch[1].to(device)
            output = model(data) 
            loss = criterion(output, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    return model



def get_probability_vectors(model, dataloader, device):
    prob_vectors = []
    model.eval()
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            outputs = model(images)
            # Get full probability distribution
            prob_dist = F.softmax(outputs, dim=1)
            prob_vectors.append(prob_dist.cpu())
    return torch.cat(prob_vectors, dim=0)

def shadow_zone(target_model, train_loader, test_loader, device='cuda'):
    in_probs = get_probability_vectors(target_model, train_loader, device)
    out_probs = get_probability_vectors(target_model, test_loader, device)
    return in_probs, out_probs

def attack_zone(target_data_col, target_label_col, target_model, in_probs, out_probs, attack_hidden_size, attack_epochs, attack_lr, device='cuda'):
    # Combine probability vectors from training and test sets
    data = torch.cat([in_probs, out_probs])
    # Create labels: 1 for training set (member), 0 for test set (non-member)
    labels = torch.tensor([1]*len(in_probs)+[0]*len(out_probs)).unsqueeze(1).to(device)
    
    dataset = TensorDataset(data, labels)
    train_loader = DataLoader(dataset, batch_size=128, shuffle=True)
    
    num_classes = in_probs.shape[1]  # Get number of classes from probability vector size
    
    # Modified attack model to accept full probability vector
    classifier = nn.Sequential(
        nn.Linear(num_classes, attack_hidden_size),
        nn.ReLU(),
        nn.Linear(attack_hidden_size, 1),
        nn.Sigmoid()
    ).to(device)
    
    criterion = nn.BCELoss()
    optimizer = optim.Adam(classifier.parameters(), lr=attack_lr)
    
    # Train the attack model
    classifier.train()
    for epoch in range(attack_epochs):
        run_loss = 0.0
        for features, labels in train_loader:
            features = features.to(device)
            labels = labels.to(device)
            
            outputs = classifier(features)
            loss = criterion(outputs, labels.float())
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            run_loss += loss.item()
            
        if epoch % 10 == 0 or epoch == attack_epochs - 1 or epoch == 0:
            print(f'Epoch {epoch+1}/{attack_epochs}, Attack Loss: {run_loss/len(train_loader):.4f}')
    
    # Get attack predictions for target data
    scores = []
    for i in range(len(target_data_col)):
        target_data = target_data_col[i]
        target_output = target_model(target_data.unsqueeze(0).to(device))
        prob_vector = F.softmax(target_output, dim=1)
        scores.append(classifier(prob_vector).item())
    
    return np.array(scores)

def MIA(target_model, train_loader, test_loader, target_data_col, target_label_col, attack_hidden_size, attack_epochs, attack_lr, device='cuda'):
    in_probs, out_probs = shadow_zone(target_model, train_loader, test_loader, device)
    scores = attack_zone(target_data_col, target_label_col, target_model, in_probs, out_probs, attack_hidden_size, attack_epochs, attack_lr, device)
    return scores