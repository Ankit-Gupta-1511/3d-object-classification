import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn
import torch.nn.functional as F
import torch.optim as optim

from utils.data_preprocessing import ModelNet10Dataset

from model.PointNet import PointNet

# Data Loaders
def get_dataloaders(root_dir, batch_size=4, n_points=1024):
    train_dataset = ModelNet10Dataset(root_dir, split='train', n_points=n_points)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    return train_loader

def train_model(model, train_loader, criterion, optimizer, num_epochs=10):
    for epoch in range(num_epochs):
        model.train()  # Set model to training mode
        total_train_loss = 0
        for data, labels in train_loader:
            data, labels = data.cuda(), labels.cuda()
            optimizer.zero_grad()  # Zero the parameter gradients
            outputs = model(data)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

        print(f'Epoch {epoch+1}, Train Loss: {total_train_loss / len(train_loader)}')


def save_model(model, path):
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")



if __name__ == '__main__':

    root_dir = 'data'
    save_path = 'output/model_weights.pth'
    batch_size = 4
    n_points = 1024
    num_epochs = 100

    model = PointNet(output_classes=10)
    if torch.cuda.is_available():
        model.cuda()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)


    train_loader = get_dataloaders(root_dir=root_dir, batch_size=batch_size, n_points=n_points)

    # Run the training loop
    train_model(model, train_loader, criterion, optimizer, num_epochs=num_epochs)
    save_model(model, save_path)