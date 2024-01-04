import os
import torch
from tqdm import tqdm

def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    save_path = f"../data/{filename}"  # Caminho para a pasta 'data' no diretório pai
    torch.save(state, save_path)

def load_checkpoint(checkpoint_path, model, optimizer):
    if os.path.isfile(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        print("Checkpoint carregado com sucesso.")
    else:
        print("Arquivo de checkpoint não encontrado.")

def train(model, criterion, optimizer, train_loader, test_loader, device, epochs):
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

        checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        }
        save_checkpoint(checkpoint, f"checkpoint_epoch_{epoch}.pth.tar")

        accuracy = evaluate(model, test_loader, device)
        print(f'Epoch {epoch+1}, Loss: {running_loss/len(train_loader)}, Accuracy: {accuracy:.2f}%')

def evaluate(model, test_loader, device):
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
