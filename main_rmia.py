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
from rmia.rmia import *
from rmia.model import *


mean = [0.5,0.5,0.5]
std = [0.5,0.5,0.5]
batch_size=128*2
channel = 3
im_size = (32, 32)
input_shape=(channel, im_size[0], im_size[1])
num_classes = 10
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
training_epochs= 50
hidden_size = 512
lr_original_model=1e-3
num_shadow_models=2
lr_shadow_model=1e-2
epochs_shadow_model=30
N_train = 20   # num of samples of train to include to testing
N_test = 20    # num of samples of test to include to testing
perc_train_in_shadow= 0.1   # perctange of training data in shadow dataset
perc_test_in_shadow= 0.8   # perctange of test data in shadow dataset
random_sample_number=100
gamma=0.5


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std)
])

dst_train = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(dst_train, batch_size=batch_size,
                                        shuffle=True, num_workers=2)

dst_test = torchvision.datasets.CIFAR10(root='./data', train=False,
                                    download=True, transform=transform)
test_loader = torch.utils.data.DataLoader(dst_test, batch_size=batch_size,
                                        shuffle=False, num_workers=2)

total_train_size = len(dst_train)
total_test_size = len(dst_test)
dst_train_residual, dst_train_shadow = torch.utils.data.random_split(dst_train, [int((1-perc_train_in_shadow)*total_train_size), total_train_size - int((1-perc_train_in_shadow)*total_train_size)])
dst_test_residual, dst_test_shadow = torch.utils.data.random_split(dst_test, [int((1-perc_test_in_shadow)*total_test_size) , total_test_size-int((1-perc_test_in_shadow)*total_test_size)])



# combine dst_train_shadow and dst_test_shadow
dst_train_shadow = torch.utils.data.ConcatDataset([dst_train_shadow, dst_test_shadow])


train_residual_loader = torch.utils.data.DataLoader(dst_train_residual, batch_size=batch_size,
                                        shuffle=True, num_workers=2)
test_residual_loader = torch.utils.data.DataLoader(dst_test_residual, batch_size=batch_size,
                                        shuffle=False, num_workers=2)

total_train_size = len(dst_train)
total_test_size = len(dst_test)
dst_train_residual, dst_train_shadow = torch.utils.data.random_split(dst_train, [int((1-perc_train_in_shadow)*total_train_size), total_train_size - int((1-perc_train_in_shadow)*total_train_size)])
dst_test_residual, dst_test_shadow = torch.utils.data.random_split(dst_test, [int((1-perc_test_in_shadow)*total_test_size) , total_test_size-int((1-perc_test_in_shadow)*total_test_size)])



# combine dst_train_shadow and dst_test_shadow
dst_train_shadow = torch.utils.data.ConcatDataset([dst_train_shadow, dst_test_shadow])


train_residual_loader = torch.utils.data.DataLoader(dst_train_residual, batch_size=batch_size,
                                        shuffle=True, num_workers=2)
test_residual_loader = torch.utils.data.DataLoader(dst_test_residual, batch_size=batch_size,
                                        shuffle=False, num_workers=2)

shadow_loader=torch.utils.data.DataLoader(dst_train_shadow, batch_size=batch_size,  
                                        shuffle=True, num_workers=2)




# train original model
net= CNN(channel, num_classes).to(device)
train_model_with_loader(net, train_residual_loader, training_epochs=training_epochs, lr=lr_original_model)
train_acc=get_accuracy(net, train_residual_loader)
test_acc=get_accuracy(net, test_residual_loader)
print(f"Train Accuracy: {train_acc}")
print(f"Test Accuracy: {test_acc}")


shadow_imgs=[]
shadow_labels=[]

for img, label in shadow_loader:
    shadow_imgs.append(img)
    shadow_labels.append(label)

shadow_imgs=torch.cat(shadow_imgs)
shadow_labels=torch.cat(shadow_labels)



train_img = next(iter(train_residual_loader))[0][0:N_train]
train_label = next(iter(train_residual_loader))[1][0:N_train]
test_img=next(iter(test_residual_loader))[0][0:N_test]
test_label=next(iter(test_residual_loader))[1][0:N_test]


target_img_coll=torch.cat((train_img, test_img))
target_label_coll=torch.cat((train_label, test_label))
reference_target=[1]*len(train_img)+[0]*len(test_img)
reference_target=np.array(reference_target)

print(f"Using {num_shadow_models} shadow models to estimate loss distributions")
print(f"Testing Attack on {N_train} train samples and {N_test} test samples")


mia_result=RMIA(target_img_coll,
            target_label_coll,
            net,
            shadow_imgs,
            shadow_labels,
            num_shadow_models,
            epochs_shadow_model,
            lr_shadow_model,
            random_sample_number,
            gamma)

mia_result=np.array(mia_result).astype(np.float32)


tpr, fpr, roc = roc_curve(reference_target, mia_result)
print('AUC:', auc(fpr, tpr))


plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % auc(fpr, tpr))
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.savefig('roc_curve.png')

