import os
import torch
from tqdm import tqdm
from sklearn.metrics import classification_report
import numpy as np

def save_checkpoint(state, filename="checkpoint.pth.tar"):
    """
    Salva o estado do treinamento atual em um arquivo de checkpoint.
    
    Args:
        state (dict): Estado do modelo e otimizador.
        filename (str): Nome do arquivo para salvar o checkpoint.
    """
    save_path = f"../data/{filename}"  # Define o caminho de salvamento no diretório 'data'
    torch.save(state, save_path)
    print("Checkpoint salvo com sucesso.")

def load_checkpoint(checkpoint_path, model, optimizer):
    """
    Carrega um checkpoint se disponível e retorna a melhor acurácia encontrada.
    
    Args:
        checkpoint_path (str): Caminho do arquivo de checkpoint.
        model (torch.nn.Module): Modelo a ser carregado.
        optimizer (torch.optim.Optimizer): Otimizador a ser carregado.
    
    Returns:
        float: A melhor acurácia registrada no checkpoint.
    """
    if os.path.isfile(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        accuracy = checkpoint.get("accuracy", 0.0)  # Recupera a acurácia, ou 0 se não existir
        print(f"Checkpoint carregado com precisão de {accuracy:.2f}%.")
        return accuracy
    else:
        print("Arquivo de checkpoint não encontrado.")
        return 0.0  # Retorna 0 se não encontrar checkpoint

def train(model, criterion, optimizer, train_loader, test_loader, device, epochs, num_classes, class_mapping, checkpoint_path):
    """
    Função para treinar o modelo.

    Args:
        model (torch.nn.Module): O modelo a ser treinado.
        criterion (torch.nn.Module): A função de perda.
        optimizer (torch.optim.Optimizer): O otimizador.
        train_loader (torch.utils.data.DataLoader): DataLoader para os dados de treino.
        test_loader (torch.utils.data.DataLoader): DataLoader para os dados de teste.
        device (torch.device): Dispositivo de computação (cpu ou cuda).
        epochs (int): Número de épocas para treinar.
        num_classes (int): Número de classes no dataset.
        class_mapping (dict): Mapeamento de classes para índices.
        checkpoint_path (str): Caminho para o arquivo de checkpoint.
    """
    best_accuracy = load_checkpoint(checkpoint_path, model, optimizer)

    for epoch in range(epochs):
        model.train()
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
            progress_bar.set_description(f"Época {epoch+1}/{epochs} - Perda: {running_loss/(i+1):.4f}")

        # Avalia o modelo após cada época
        model.eval()
        all_predictions = []
        all_labels = []

        with torch.no_grad():
            for images, labels in tqdm(test_loader, desc="Avaliando"):
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        # Calcula a precisão global e gera o relatório de classificação
        current_accuracy = (np.array(all_predictions) == np.array(all_labels)).mean()
        class_names_ordered = [class_mapping[i] for i in sorted(class_mapping)]
        report = classification_report(all_labels, all_predictions, target_names=class_names_ordered)
        print("\nRelatório de Classificação:\n", report)

        # Salva o checkpoint se a precisão atual for maior que a melhor precisão
        if current_accuracy > best_accuracy:
            best_accuracy = current_accuracy
            checkpoint = {
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "accuracy": best_accuracy,
            }
            save_checkpoint(checkpoint, "checkpoint.pth.tar")

        print(f'Época {epoch+1} concluída, Perda: {running_loss/len(train_loader)}, Precisão: {current_accuracy:.2f}%')

