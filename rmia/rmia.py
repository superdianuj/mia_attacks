import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import transforms
import numpy as np
import torch.nn.functional as F
from scipy.stats import norm
from sklearn.metrics import roc_curve, auc, accuracy_score
import math
import matplotlib.pyplot as plt 
from rmia.shadow import *




def get_accuracy(model, data_loader, device='cuda'):
    correct = 0
    total = 0
    model.eval()
    with torch.no_grad():
        for data in data_loader:
            images, labels = data
            outputs = model(images.to(device))
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels.to(device)).sum().item()
    return correct / total*100



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


# Function to train a model
def train_model_with_raw_tensors(model, train_data, train_labels, epochs=100, lr=0.01,bs=128*2, device='cuda'):
    dataset= torch.utils.data.TensorDataset(train_data, train_labels)
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=bs, shuffle=True)
    criterion = nn.CrossEntropyLoss()
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=lr)
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


def get_prob(model,data,target,device):
    model.eval()
    lab=target[0].item()
    with torch.no_grad():
        output = model(data.to(device))
        softmax_output = nn.Softmax(dim=1)(output)
        prob = softmax_output[0][lab].detach().cpu().item()
    return prob




def shadow_zone(
                shadow_data,
                shadow_labels,
                target_data,
                target_labels,
                num_shadow_models=2,
                shaodw_epochs=30,
                shadow_lr=1e-2,
                random_sample_number=100,
                device='cuda'
                ):
    

    rand_indices = torch.randperm(shadow_data.size(0))
    random_data = shadow_data[rand_indices[:random_sample_number]]
    random_labs = shadow_labels[rand_indices[:random_sample_number]]

    in_probs = [[] for _ in range(num_shadow_models)]
    out_probs = [[] for _ in range(num_shadow_models)]
    rand_probs_per_target=[]
    for omeaga in range(target_data.size(0)):
        rand_probs=[[] for _ in range(num_shadow_models)]
        for ind in range(num_shadow_models):

            # split the shadow_data and shadow_labs into two parts (random)
            indices = torch.randperm(shadow_data.size(0))
            shadow_data1 = shadow_data[indices[:len(indices)//2]]
            shadow_labs1 = shadow_labels[indices[:len(indices)//2]]
            shadow_data2 = shadow_data[indices[len(indices)//2:]]
            shadow_labs2 = shadow_labels[indices[len(indices)//2:]]

            # Include the target example in the dataset
            model_in = ShadowModel().to(device)
            train_data_in = torch.cat((shadow_data1, target_data[omeaga:omeaga+1]), 0)
            train_labels_in = torch.cat((shadow_labs1, target_labels[omeaga:omeaga+1]), 0)
            model_in = train_model_with_raw_tensors(model_in, train_data_in, train_labels_in, shaodw_epochs, shadow_lr)
            prob_in=get_prob(model_in,target_data[omeaga:omeaga+1],target_labels[omeaga:omeaga+1],device)
            in_probs[ind].append(prob_in)

            # Exclude the target example from the dataset
            model_out = ShadowModel().to(device)
            train_data_out=shadow_data2
            train_labels_out = shadow_labs2
            model_out = train_model_with_raw_tensors(model_out, train_data_out, train_labels_out, shaodw_epochs, shadow_lr)
            prob_out=get_prob(model_out,target_data[omeaga:omeaga+1],target_labels[omeaga:omeaga+1],device)
            out_probs[ind].append(prob_out)

            # get probs on random data
            for i in range(random_sample_number):
                prob_random=get_prob(model_out,random_data[i:i+1],random_labs[i:i+1],device)
                rand_probs[ind].append(prob_random)


        rand_probs=np.array(rand_probs)   # (shadow_models-out, random_samples)
        # take mean along shadow_models(out)
        rand_probs_over_shadow_out=np.mean(rand_probs,axis=0)  # (random_samples)
        rand_probs_per_target.append(rand_probs_over_shadow_out)   

    in_probs=np.array(in_probs)     # (shadow_models-in, target_samples)
    out_probs=np.array(out_probs)   # (shadow_models-out, target_samples)
    rand_probs_per_target=np.array(rand_probs_per_target)  # (target_samples, random_samples)

    # take mean along shadow_models(in)
    in_probs=np.mean(in_probs,axis=0)  # (target_samples)
    out_probs=np.mean(out_probs,axis=0)  # (target_samples)

    prob_target=1/2*(in_probs+out_probs)   # (target_samples)

    return random_data,random_labs,prob_target,rand_probs_per_target



def attack_zone(target_model,
                target_data,
                target_labels,
                random_data,
                random_labs,
                prob_target,
                rand_probs_per_target,
                gamma=0.5,device):
    scores=[]
    for i in range(target_data.size(0)):
        target_prob=prob_target[i]
        target_prob_given_target_model=get_prob(target_model,target_data[i:i+1],target_labels[i:i+1],device)
        lr_target=target_prob_given_target_model/(target_prob_given_target_model+1e-15)
        C=0
        for j in range(random_data.size(0)):
            rand_prob=rand_probs_per_target[i][j]
            rand_prob_given_target_model=get_prob(target_model,random_data[j:j+1],random_labs[j:j+1],device)
            lr_rand=rand_prob_given_target_model/(rand_prob_given_target_model+1e-15)
            C+=1 if lr_target/lr_rand>gamma else 0

        scores.append(C/random_data.size(0))

    return np.array(scores)






def RMIA(target_data,
         target_labels,
         target_model,
            shadow_data,
            shadow_labels,
            num_shadow_models=2,
            shaodw_epochs=30,
            shadow_lr=1e-2,
            random_sample_number=100,
            gamma=0.5, device='cuda'):
    

    random_data,random_labs,prob_target,rand_probs_per_target=shadow_zone(shadow_data,shadow_labels,target_data,target_labels,num_shadow_models,shaodw_epochs,shadow_lr,random_sample_number,device)
    scores=attack_zone(target_model,target_data,target_labels,random_data,random_labs,prob_target,rand_probs_per_target,gamma,device)
    return scores
