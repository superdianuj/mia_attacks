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
from shadow_attack.shadow_attack import *
from shadow_attack.shadow import *
from shadow_attack.model import *



# Main function
if __name__ == "__main__":
    TRAIN_RS=[0.01, 0.05, 0.1, 0.2,0.3]
    AUC=[]

    for train_r in TRAIN_RS:
        num_classes = 10
        num_epochs = 50
        nshadow_num_epochs = 30
        batch_size = 128
        learning_rate = 0.001
        test_r = 0.9
        num_shadow_models = 5

        # Attack Model Hyperparameters
        num_attack_epochs = 100
        attack_batch_size = 10
        attack_learning_rate = 0.001

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Load data
        target_loader, shadow_loaders, test_loader, shadow_loader_test = get_data_loaders(batch_size, train_r, test_r, num_shadow_models)
        print("======================================================")
        print("Attackers Knowledge:")
        print(f"Using {num_shadow_models} shadow models")
        print(f"Picking {train_r*100}% of the training data for each shadow model: {len(shadow_loaders[0].dataset)}/{len(target_loader.dataset)}")
        print(f"Picking {test_r*100}% of the test data for the shadow models: {len(shadow_loader_test.dataset)}/{len(test_loader.dataset)}")
        print("======================================================")
        
        input_size = 3 * 32 * 32
        hidden_size = 128*3
        channel = 3
        criterion = nn.CrossEntropyLoss()

        # Train target model
        if not os.path.exists('target_model.pth'):
            target_model = CNN(channel, num_classes).to(device)
            optimizer = optim.Adam(target_model.parameters(), lr=learning_rate)
            train_model(target_model, target_loader, criterion, optimizer, num_epochs, device)
            torch.save(target_model.state_dict(), 'target_model.pth')
        else:
            target_model = CNN(channel, num_classes).to(device)
            target_model.load_state_dict(torch.load('target_model.pth'))

        print('Target Model Accuracy on Training Data: ', get_model_accuracy(target_model, target_loader, device))
        print('Target Model Accuracy on Test Data: ', get_model_accuracy(target_model, test_loader, device))

        # Train shadow models
        shadow_models = []
        for i in range(num_shadow_models):
            shadow_model = ShadowModel(channel, num_classes).to(device)
            optimizer = optim.Adam(shadow_model.parameters(), lr=learning_rate)
            print(f'\nTraining Shadow Model {i+1}/{num_shadow_models}')
            train_model(shadow_model, shadow_loaders[i], criterion, optimizer, nshadow_num_epochs, device)
            shadow_models.append(shadow_model)

        attack_loader = create_attack_loader(shadow_models, target_model, shadow_loaders, shadow_loader_test, attack_batch_size)

        # Train attack model
        attack_model = MLP(num_classes, hidden_size, 2).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(attack_model.parameters(), lr=attack_learning_rate)
        print('\nTraining Attack Model')
        train_attack_model(attack_model, attack_loader, criterion, optimizer, num_attack_epochs, device)

        # Evaluate attack model
        print('Evaluating attack model on training data')
        print('----------------------------------------------------------')
        auc_train=evaluate_attack_model(attack_model, attack_loader, device)
        print('----------------------------------------------------------')
        print('\n')
        print('__________________________________________________________')
        print('||______________________________________________________||')

        # evaluate on test data
        attack_loader_test = create_attack_loader(shadow_models, target_model, [target_loader]*num_shadow_models, test_loader, attack_batch_size)
        print('Evaluating attack model on test data')
        print('----------------------------------------------------------')
        auc_test=evaluate_attack_model(attack_model, attack_loader_test, device)
        print('----------------------------------------------------------')

        AUC.append(auc_test)






    TRAIN_RS=[r*100 for r in TRAIN_RS]
    if os.path.exists('shadowMIA_multi_shadow'):
        os.remove('shadowMIA_multi_shadow')

    os.mkdir('shadowMIA_multi_shadow')

    sns.set_style("whitegrid")
    plt.figure(figsize=(10, 5), dpi=150)
    plt.plot(TRAIN_RS, AUC,marker='o',label='Attack AUC over whole CIFAR-10 Dataset')
    plt.xlabel('Attackers Knowledge (Amount of Training Data Known) %')
    plt.ylabel('Membership Inference Attack-AUC')
    plt.legend()
    plt.ylim(0, 1)  # Set y-axis from 0 to 1
    plt.yticks([i * 0.05 for i in range(21)])  # Create y-ticks from 0 to 1 with 0.05 intervals
    plt.grid(True, which='both', axis='y', linestyle='--', linewidth=0.5)  # Add horizontal grid lines
    plt.title('Membership Inference Recall vs Percentage of Attackers Knowledge')
    plt.savefig('shadowMIA_multi_shadow/Membership_Inference_Recall_vs_Percentage_of_Attackers_Knowledge.png')



