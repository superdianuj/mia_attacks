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
from shadow import *





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


def stable_phi(loss):
    loss=max(0.0,loss)
    exp_term = math.exp(-loss)+1e-15
    phi = np.log(exp_term) - np.log1p(-exp_term)
    return phi


# Function to estimate loss distributions using shadow models
def estimate_loss_distributions(target_data, target_label, shadow_imgs,shadow_labs, num_shadow_models=2, epochs=100, lr=0.01, device='cuda'):
    in_losses = [[] for _ in range(target_data.size(0))]
    out_losses = [[] for _ in range(target_data.size(0))]

    for omeaga in range(target_data.size(0)):
        for ind in range(num_shadow_models):

            # split the shadow_imgs and shadow_labs into two parts (random)
            indices = torch.randperm(shadow_imgs.size(0))
            shadow_imgs1 = shadow_imgs[indices[:len(indices)//2]]
            shadow_labs1 = shadow_labs[indices[:len(indices)//2]]
            shadow_imgs2 = shadow_imgs[indices[len(indices)//2:]]
            shadow_labs2 = shadow_labs[indices[len(indices)//2:]]

            # Include the target example in the dataset
            model_in = ShadowModel().to(device)
            train_data_in = torch.cat((shadow_imgs1, target_data[omeaga:omeaga+1]), 0)
            train_labels_in = torch.cat((shadow_labs1, target_label[omeaga:omeaga+1]), 0)
            model_in = train_model_with_raw_tensors(model_in, train_data_in, train_labels_in, epochs, lr)
            model_in.eval()
            with torch.no_grad():
                output_in = model_in(target_data[omeaga:omeaga+1].to(device))
                loss_in = nn.CrossEntropyLoss()(output_in, target_label[omeaga:omeaga+1].to(device)).cpu().numpy()
                in_losses[omeaga].append(stable_phi(loss_in))

            # Exclude the target example from the dataset
            model_out = ShadowModel().to(device)
            train_data_out=shadow_imgs2
            train_labels_out = shadow_labs2
            model_out = train_model_with_raw_tensors(model_out, train_data_out, train_labels_out, epochs, lr)
            model_out.eval()
            with torch.no_grad():
                output_out = model_out(target_data[omeaga:omeaga+1].to(device))
                loss_out = nn.CrossEntropyLoss()(output_out, target_label[omeaga:omeaga+1].to(device)).cpu().numpy()
                out_losses[omeaga].append(stable_phi(loss_out))


    in_mean_list = [np.mean(losses) for losses in in_losses]
    out_mean_list = [np.mean(losses) for losses in out_losses]
    in_std_list = [np.std(losses) for losses in in_losses]
    out_std_list = [np.std(losses) for losses in out_losses]

    return in_mean_list, in_std_list, out_mean_list, out_std_list


# Function to perform membership inference
def membership_inference(model, target_data, target_label, in_mean, in_std, out_mean, out_std, device='cuda'):
    model.eval()
    with torch.no_grad():
        output = model(target_data.to(device))
        loss = nn.CrossEntropyLoss()(output, target_label.to(device)).item()
        loss=stable_phi(loss)

    # Calculate likelihoods assuming Gaussian distributions
    #------Version-1-------------------------------------------------------------------------------------
    p_in = (1 / (in_std * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((loss - in_mean) / in_std) ** 2)
    p_out = (1 / (out_std * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((loss - out_mean) / out_std) ** 2)

    return p_in/(p_out+1e-15)
    #----------------------------------------------------------------------------------------------------

    #-----Version-2--------------------------------------------------------------------------------------
    # p_in = norm.pdf(loss, in_mean, in_std)
    # p_out = norm.pdf(loss, out_mean, out_std)

    # return p_in/(p_out+1e-15)
    #----------------------------------------------------------------------------------------------------

    #------Version-3-------------------------------------------------------------------------------------
    # p_in = -norm.logpdf(loss,in_mean, in_std + 1e-30)
    # p_out = -norm.logpdf(loss, out_mean, out_std + 1e-30)
    # return p_in - p_out
    #----------------------------------------------------------------------------------------------------


def run_over_MIA(model,target_data_col,target_label_col,in_mean_col,in_std_col,out_mean_col,out_std_col):
    result_col=[]
    for i in range(len(target_data_col)):
        result = membership_inference(model, target_data_col[i:i+1], target_label_col[i:i+1], in_mean_col[i], in_std_col[i], out_mean_col[i], out_std_col[i])
        result_col.append(result)
    return result_col

