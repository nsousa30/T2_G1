#!/usr/bin/env python3

import os
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torch.optim import Adam
from torch import nn
from sklearn.model_selection import train_test_split
from dataset import RGBDDataset
from model import SimpleCNN
from trainer import train, evaluate, save_checkpoint, load_checkpoint

# Configurações iniciais
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

dataset_path = '../data/rgbd-dataset'

epochs = 2

checkpoint_path = "../data/my_checkpoint.pth.tar"

# Transformações
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Redimensionar as imagens para 224x224
    transforms.ToTensor(),          # Converter para tensor
])

# Carregar o dataset
dataset = RGBDDataset(dataset_path, transform=transform)

# Divisão de dados em treino e teste
train_indices, test_indices = train_test_split(range(len(dataset)), test_size=0.2)
train_data = Subset(dataset, train_indices)
test_data = Subset(dataset, test_indices)

# DataLoaders
train_loader = DataLoader(train_data, batch_size=64, shuffle=True)  # Tamanho do lote reduzido
test_loader = DataLoader(test_data, batch_size=64, shuffle=False)

# Modelo
num_classes = len(os.listdir(dataset_path))
model = SimpleCNN(num_classes=num_classes).to(device)

# Critério e Otimizador
criterion = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=1e-4)

if not os.path.exists('../data'):
    os.makedirs('../data')

if os.path.exists(checkpoint_path):
    load_checkpoint(checkpoint_path, model, optimizer)


# Treinamento
train(model, criterion, optimizer, train_loader, test_loader, device, epochs)
