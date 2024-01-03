#!/usr/bin/env python3

from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from dataset import RGBDDataset

# Transformações que você pode querer aplicar
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Criando o dataset
dataset = RGBDDataset('../rgbd-dataset', transform=transform)

# Obtenha os índices de todos os dados
indices = list(range(len(dataset)))

# Divida os índices em índices de treino e teste
train_indices, test_indices = train_test_split(indices, test_size=0.2)

# Crie subconjuntos do dataset usando os índices de treino e teste
from torch.utils.data import Subset
train_data = Subset(dataset, train_indices)
test_data = Subset(dataset, test_indices)

# Criando DataLoaders
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
test_loader = DataLoader(test_data, batch_size=32, shuffle=False)


