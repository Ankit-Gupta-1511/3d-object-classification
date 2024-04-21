
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torchviz import make_dot

from utils.data_preprocessing import ModelNet10Dataset

from model.PointNetSequential import PointNet

# Data Loaders
def get_dataloaders(root_dir, batch_size=4, n_points=1024):
    train_dataset = ModelNet10Dataset(root_dir, split='test', n_points=n_points)

    test_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    return test_loader

def eval_model(model, test_loader, criterion, optimizer):
        # Validation phase
        model.eval()  # Set model to evaluate mode
        total_val_loss = 0
        correct = 0
        with torch.no_grad():
            for data, labels in test_loader:
                data, labels = data.cuda(), labels.cuda()
                outputs = model(data)
                loss = criterion(outputs, labels)
                total_val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                print(f' Loss: {loss.item()}, Correct: {(predicted == labels).sum().item()}')
                correct += (predicted == labels).sum().item()

        print(f' Val Loss: {total_val_loss / len(test_loader)}, Accuracy: {correct / len(test_loader.dataset)}')
        dot = make_dot(outputs, params=dict(model.named_parameters()))

        dot.render('output/model_architecture', format='png')


def load_model(model, path):
    model.load_state_dict(torch.load(path))
    print(f"Model loaded from {path}")



if __name__ == '__main__':

    root_dir = 'data'
    save_path = 'output/model_weights.pth'
    batch_size = 4
    n_points = 1024
    num_epochs = 10

    model = PointNet(output_classes=10)
    if torch.cuda.is_available():
        model.cuda()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    test_loader = get_dataloaders(root_dir=root_dir, batch_size=batch_size, n_points=n_points)

    load_model(model, save_path)
    # Run the training loop
    eval_model(model, test_loader, criterion, optimizer)

    



