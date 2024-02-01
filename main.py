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
from trainer import train, load_checkpoint
from PIL import Image
from torchvision.transforms import Compose, ToTensor
import torch.nn.functional as F


def classify_loaded_image(image, model, class_mapping, device, transform):
    """
    Classifica uma imagem previamente carregada.

    :param image: Objeto de imagem PIL já carregado.
    :param model: Modelo treinado para classificação.
    :param class_mapping: Dicionário que mapeia índices de classes para nomes de classes.
    :param device: Dispositivo no qual o modelo está a correr, por exemplo 'cuda' ou 'cpu'.
    :return: Nome da classe prevista.
    """
   
    # Aplicar as transformações na imagem
    image_tensor = transform(image).unsqueeze(0)  # Adiciona uma dimensão de lote
    #image_tensor = transform(image)  # Adiciona uma dimensão de lote
    image_tensor = image_tensor.to(device)

    # Colocar o modelo em modo de avaliação e fazer a previsão
    model.eval()
    with torch.no_grad():
        outputs = model(image_tensor)
        predicted_probabilities = F.softmax(outputs, dim=1).tolist()
        print("\npredicted_probabilities")
        print(predicted_probabilities)
        _, predicted_idx = torch.max(outputs, 1)

    # Mapear o índice previsto de volta para o nome da classe original
    predicted_class = class_mapping[predicted_idx.item()]

    return predicted_class


if __name__ == "__main__":

    # Define se o dispositivo de treinamento será CPU ou GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Usando o dispositivo: {device}")

    # Define o caminho para o dataset e o caminho para guardar os checkpoints
    dataset_path = './rgbd-dataset'
    # checkpoint_path = "../data/checkpoint.pth.tar"
    checkpoint_path = "../trained_model/checkpoint.pth.tar"

    # Define o número de épocas do treino
    epochs = 10

    # Define as transformações aplicadas nas imagens
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Redimensiona as imagens para 224x224
        transforms.ToTensor(),          # Converte as imagens para tensores
    ])

    # Carrega o dataset
    dataset = RGBDDataset(dataset_path, transform=transform)

    # Divide o dataset em conjuntos de treino e teste
    train_indices, test_indices = train_test_split(range(len(dataset)), test_size=0.2)
    train_data = Subset(dataset, train_indices)
    test_data = Subset(dataset, test_indices)

    # Configura os DataLoaders para treino e teste
    train_loader = DataLoader(train_data, batch_size=100, shuffle=True, num_workers=12)
    test_loader = DataLoader(test_data, batch_size=100, shuffle=False, num_workers=12)

    # Cria o modelo e define o critério e otimizador
    num_classes = len(os.listdir(dataset_path))
    model = SimpleCNN(num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=1e-4)

    # # Se não existir a pasta para salvar os dados, cria-a
    if not os.path.exists('../trained_model'):
         os.makedirs('../trained_model')

    # Se existir a pasta, tenta carregar o checkpoint
    if os.path.exists('../trained_model'):
        load_checkpoint(checkpoint_path, model, optimizer)

    # Obtém o mapeamento de classes
    class_mapping = dataset.get_class_mapping()

    # Inicia o treino------------------------------------------------------------------------------------------------------------------
    #train(model, criterion, optimizer, train_loader, test_loader, device, epochs, num_classes, class_mapping, checkpoint_path)

    # OU

    # Inicia o teste--------------------------------------------------------------------------------------------------------------------
    imagem = Image.open("soda_can_6_1_66_crop.png")
    imagem.show()

    prediction = classify_loaded_image(imagem, model=model, class_mapping=class_mapping, device=device, transform=transform)

    print("\nPrediction:")
    print(prediction)