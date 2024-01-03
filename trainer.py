# Função de treinamento
from torch import device
from tqdm import tqdm
import torch


def train(model, criterion, optimizer, train_loader, test_loader, epochs=2, device='cuda'):
    for epoch in range(epochs):
        model.train()  # Coloca o modelo em modo de treinamento
        running_loss = 0.0
        progress_bar = tqdm(enumerate(train_loader), total=len(train_loader))

        for i, (images, labels) in progress_bar:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            progress_bar.set_description(f"Epoch {epoch+1}/{epochs} Loss: {running_loss/(i+1):.4f}")

        # Avaliação após cada época
        accuracy = evaluate(model, test_loader)
        print(f'Epoch {epoch+1}, Loss: {running_loss/len(train_loader)}, Accuracy: {accuracy:.2f}%')

def evaluate(model, test_loader):
    model.eval()  # Coloca o modelo em modo de avaliação
    correct = 0
    total = 0

    with torch.no_grad():  # Desativa o cálculo do gradiente
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    return accuracy