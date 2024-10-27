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
from sklearn.metrics import roc_curve, auc, accuracy_score
from torchvision import datasets, transforms
from shadow_attack.shadow_attack import *
from shadow_attack.shadow import *
from shadow_attack.model import *

# -----------------------------------------------------------------------------------
input_shape = (3, 32, 32)
channel = 3
num_classes=10
hidden_size = 512
output_size = 10
epochs = 50
lr = 1e-3
perc=0.0   # amount of actual training data available to the attacker
perc_test=0.20    # amount of testing data available to the attacker ( similar distribution to training data)
meausurement_number=10  # number of target samples to be measured from each training and non-training data
num_shadow_models=5
lr_shadow_model=1e-3
epochs_shadow_model=30
lr_attack_model=1e-3
epochs_attack_model=50
attack_hidden_size=128
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#-----------------------------------------------------------------------------------

if not os.path.exists('results'):
    os.makedirs('results')

# Load CIFAR-10 dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    transforms.Lambda(lambda x: x.view(3, 32, 32))
])

train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)


batch_size = 128 * 2
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


target_model=CNN(channel, num_classes).to(device)
if not os.path.exists('target_model.pth'):
    print(f"Training Target Model on CIFAR-10 on Epochs: {epochs}")
    train_model_with_loader(target_model, train_loader, epochs, lr,device)
    torch.save(target_model.state_dict(), 'target_model.pth')

else:
    print("Loading trained Target Model")
    target_model.load_state_dict(torch.load('target_model.pth'))
    target_model.to(device)

# Calculate training and test accuracy
train_accuracy = calculate_accuracy(target_model, train_loader, device)
test_accuracy = calculate_accuracy(target_model, test_loader, device)
print(f"Training Accuracy: {train_accuracy:.2f}%")
print(f"Test Accuracy: {test_accuracy:.2f}%")


num_samples_train = int(perc*len(train_loader.dataset))
num_samples_test=int(perc_test*len(test_loader.dataset))
print("----------------------------------")
print(f"Attackers knowledge:")
print(f"Training Dataset Info: {num_samples_train}/{len(train_loader.dataset)} = {num_samples_train/len(train_loader.dataset)*100}%")
print(f"Testing Dataset Info: {num_samples_test}/{len(test_loader.dataset)} = {num_samples_test/len(test_loader.dataset)*100}%")
print("----------------------------------")


train_images=[]
train_labels=[]
for images, labels in train_loader:
    train_images.append(images)
    train_labels.append(labels)
train_images=torch.cat(train_images)
train_labels=torch.cat(train_labels)

if not os.path.exists('original_indices'):
    original_indices=torch.randperm(len(train_images))
    torch.save(original_indices,'original_indices')
else:
    original_indices=torch.load('original_indices')

indices = original_indices[:num_samples_train]
anti_indices = original_indices[num_samples_train:num_samples_train+meausurement_number]
attacker_train_images = train_images[indices]
attacker_train_labels = train_labels[indices]

measurement_train_images = train_images[anti_indices]
measurement_train_labels = train_labels[anti_indices]


test_images=[]
test_labels=[]
for images, labels in test_loader:
    test_images.append(images)
    test_labels.append(labels)
test_images=torch.cat(test_images)
test_labels=torch.cat(test_labels)

if not os.path.exists('original_indices_test'):
    original_indices_test=torch.randperm(len(test_images))
    torch.save(original_indices_test,'original_indices_test')
else:
    original_indices_test=torch.load('original_indices_test')


indices = original_indices_test[:num_samples_test]
anti_indices = original_indices_test[num_samples_test:num_samples_test+meausurement_number]


attacker_test_images = test_images[indices]
attacker_test_labels = test_labels[indices]

measurement_test_images = test_images[anti_indices]
measurement_test_labels = test_labels[anti_indices]

shadow_images=torch.cat([attacker_train_images,attacker_test_images])
shadow_labels=torch.cat([attacker_train_labels,attacker_test_labels])


measurement_images=torch.cat([measurement_train_images,measurement_test_images])
measurement_ref=np.array([0]*len(measurement_train_images)+[1]*len(measurement_test_images))
measurement_labels=torch.cat([measurement_train_labels,measurement_test_labels])

print("Measurement Sample Size:",len(measurement_images))


scores=run_over_MIA(target_model, 
                    measurement_images, 
                    shadow_images, 
                    shadow_labels, 
                    num_shadow_models, 
                    epochs_shadow_model, 
                    lr_shadow_model,  
                    epochs_attack_model, 
                    lr_attack_model, 
                    attack_hidden_size, 
                    device)



tpr, fpr, roc = roc_curve(measurement_ref, scores)
print("--------------")
print(f'AUC: {auc(fpr, tpr)}  |')
print("-------------")
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % auc(fpr, tpr))
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.savefig(f'results/ROC_Shadow.png')




filename = "results/Shadow_Attack.txt"  

hyperparams = {
    "input_shape": input_shape,
    "channel": channel,
    "num_classes": num_classes,
    "hidden_size": hidden_size,
    "output_size": output_size,
    "epochs": epochs,
    "lr": lr,
    "perc": perc,
    "perc_test": perc_test,
    "measurement_number": meausurement_number,
    "num_shadow_models": num_shadow_models,
    "lr_shadow_model": lr_shadow_model,
    "epochs_shadow_model": epochs_shadow_model,
    "lr_attack_model": lr_attack_model,
    "epochs_attack_model": epochs_attack_model,
    "attack_hidden_size": attack_hidden_size,
    "device": str(device)
}


os.makedirs("results", exist_ok=True)

with open(filename, 'w') as f:
    for param_name, param_value in hyperparams.items():
        f.write(f"{param_name}: {param_value}\n")