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
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torchmetrics.utilities.data import to_onehot
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import roc_curve, auc, accuracy_score
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset, TensorDataset
# import resnet18 from torchvision.models
import torchvision.models as models



class FeatureDataset(Dataset):
    def __init__(self, features):
        self.features = features

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.features[idx]
    

class LogitDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
    

def calculate_metrics(true_labels, predictions, positive_label=1):
    # Compute TPR, FPR, and AUC
    fpr, tpr, thresholds = roc_curve(true_labels, predictions, pos_label=positive_label)
    roc_auc = auc(fpr, tpr)
    
    return fpr, tpr, roc_auc




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






# Helper functions
def pinball_loss_fn(score, target, quantile):
    target = target.reshape([-1, 1])
    delta_score = target - score
    loss = torch.nn.functional.relu(delta_score) * quantile + torch.nn.functional.relu(-delta_score) * (1.0 - quantile)
    return loss

def gaussian_loss_fn(score, target, quantile):
    mu = score[:, 0]
    log_std = score[:, 1]
    loss = log_std + 0.5 * torch.exp(-2 * log_std) * (target - mu) ** 2
    return loss


def quantile_regression_scoring(base_classifier, target_sample, target_label, auxiliary_dataset, 
                                n_quantile=100, low_quantile=0.01, high_quantile=0.99, 
                                use_logscale=False, use_gaussian=False, batch_size=32, 
                                num_epochs=10, learning_rate=0.001, alpha=0.05, device='cpu'):
    # Set up quantiles
    if use_logscale:
        log_low = np.log10(low_quantile)
        log_high = np.log10(high_quantile)
        QUANTILE = torch.sort(
            torch.logspace(log_low, log_high, n_quantile)
        )[0].reshape([1, -1])
    else:
        QUANTILE = torch.linspace(low_quantile, high_quantile, n_quantile).reshape([1, -1])

    # Define loss function
    if use_gaussian:
        loss_fn = gaussian_loss_fn
    else:
        loss_fn = pinball_loss_fn

    # Define target scoring function
    def target_scoring_fn(samples, labels, base_classifier):
        base_classifier.eval()
        with torch.no_grad():
            logits = base_classifier(samples)
            oh_label = torch.nn.functional.one_hot(labels, num_classes=logits.shape[-1]).bool()
            score = logits[oh_label]
            # Mask out the true label before taking max over incorrect labels
            logits_masked = logits.masked_fill(oh_label, float('-inf'))
            score -= torch.max(logits_masked, dim=1)[0]
        return score, logits

    # Step 1: Evaluate the base classifier on the auxiliary dataset
    base_classifier.eval()
    features = []
    target_scores = []
    
    with torch.no_grad():
        for x, y in auxiliary_dataset:
            features.append(x)
            target_score, _ = target_scoring_fn(x, y, base_classifier)
            target_scores.append(target_score)
    
    features = torch.cat(features)
    target_scores = torch.cat(target_scores)
    
    # Create the quantile regression model
    class QuantileRegressionModel(nn.Module):
        def __init__(self, input_dim, output_dim):
            super().__init__()
            self.network = models.resnet18(pretrained=False)
            num_ftrs = self.network.fc.in_features
            self.network.fc = nn.Linear(num_ftrs, output_dim)
        
        def forward(self, x):
            return self.network(x)
        
    input_dim =   (target_sample.shape[0], target_sample.shape[1], target_sample.shape[2])
    output_dim = 2 if use_gaussian else n_quantile
    quantile_model = QuantileRegressionModel(input_dim, output_dim).to(device)
    
    # Step 2: Train the quantile regression model
    optimizer = torch.optim.Adam(quantile_model.parameters(), lr=learning_rate)
    
    train_loader = DataLoader(TensorDataset(features, target_scores), batch_size=batch_size, shuffle=True)
    
    # Training loop
    for epoch in range(num_epochs):
        quantile_model.train()
        for batch_features, batch_target_scores in train_loader:
            batch_features = batch_features.to(device)
            batch_target_scores = batch_target_scores.to(device)
            optimizer.zero_grad()
            scores = quantile_model(batch_features)
            loss = loss_fn(scores, batch_target_scores, QUANTILE.to(scores.device)).mean()
            loss.backward()
            optimizer.step()
        
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}")

    quantile_model.to('cpu')
    
    # Step 3: Compute the score for the target sample
    quantile_model.eval()
    base_classifier.eval()
    with torch.no_grad():
        target_score, _ = target_scoring_fn(target_sample.unsqueeze(0), torch.tensor([target_label]), base_classifier)
        predicted_scores = quantile_model(target_sample.unsqueeze(0))
        
        if use_gaussian:
            mu = predicted_scores[:, 0]
            log_std = predicted_scores[:, 1]
            predicted_scores = mu.reshape([-1, 1]) + torch.exp(log_std).reshape([-1, 1]) * torch.erfinv(2 * QUANTILE.to(predicted_scores.device) - 1).reshape([1, -1]) * math.sqrt(2)
        
        # Find the index of quantile closest to 1 - alpha
        quantile_value = 1 - alpha
        quantile_index = torch.argmin(torch.abs(QUANTILE - quantile_value))

        ## Compute the score (difference between target score and predicted quantile at 1 - alpha)
        score = target_score.item() - predicted_scores[0, quantile_index].item()

    return score



def mia_attack(base_classifier, target_samples, target_labels, auxiliary_dataset, 
                n_quantile=100, low_quantile=0.01, high_quantile=0.99, 
                use_logscale=False, use_gaussian=False, batch_size=32, 
                num_epochs=10, learning_rate=0.001, alpha=0.05, device='cpu'):

    scores=[]
    for i in range(len(target_samples)):
        score = quantile_regression_scoring(base_classifier, target_samples[i], target_labels[i], auxiliary_dataset, 
                                            n_quantile, low_quantile, high_quantile, 
                                            use_logscale, use_gaussian, batch_size, 
                                            num_epochs, learning_rate, alpha,device)
        print(f"Sample {i+1}/{len(target_samples)}: Score = {score:.4f}")
        scores.append(score)

    return np.array(scores)
    