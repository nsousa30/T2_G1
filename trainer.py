import os
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt

global_accuracy_plot = None
class_accuracy_plots = {}

def save_checkpoint(state, filename="checkpoint.pth.tar"):
    save_path = f"../data/{filename}"  # Caminho para a pasta 'data' no diretório pai
    torch.save(state, save_path)
    print("Checkpoint saved")

def load_checkpoint(checkpoint_path, model, optimizer):
    if os.path.isfile(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        accuracy = checkpoint.get("accuracy", 0.0)  # Pega a accuracy, ou 0 se não estiver no checkpoint
        print(f"Checkpoint carregado com sucesso com precisão de {accuracy:.2f}%.")
        return accuracy
    else:
        print("Arquivo de checkpoint não encontrado.")
        return 0.0  # Retorna 0 se nenhum checkpoint foi encontrado

def train(model, criterion, optimizer, train_loader, test_loader, device, epochs, num_classes, class_mapping, checkpoint_path):
    best_accuracy = load_checkpoint(checkpoint_path, model, optimizer)
    global_accuracies = []  # Lista para armazenar a precisão global de cada época
    class_accuracies = {class_name: [] for class_name in class_mapping.values()}  # Dicionário para precisões por classe

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

        accuracy_classes, current_accuracy = evaluate(model, test_loader, device, class_mapping)

        global_accuracies.append(current_accuracy)

        for class_name, accuracy in accuracy_classes.items():
            class_accuracies[class_name].append(accuracy)

        # Plotagem no final de cada época
        plot_accuracies(global_accuracies, class_accuracies, epoch)

        # Salva o checkpoint se a precisão atual for maior que a melhor precisão
        if current_accuracy > best_accuracy:
            best_accuracy = current_accuracy
            checkpoint = {
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "accuracy": best_accuracy,
            }
            save_checkpoint(checkpoint, f"checkpoint_epoch_{epoch}.pth.tar")

        print(f'Epoch {epoch+1}, Loss: {running_loss/len(train_loader)}, Accuracy: {current_accuracy:.2f}%')

def evaluate(model, test_loader, device, class_mapping):
    model.eval()
    class_correct = list(0. for _ in class_mapping)
    class_total = list(0. for _ in class_mapping)

    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Avaliando"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            c = (predicted == labels).squeeze()
            for i in range(len(labels)):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1

    accuracies_per_class = {}
    for i, class_name in class_mapping.items():
        accuracy = 100 * class_correct[i] / class_total[i] if class_total[i] != 0 else 0
        accuracies_per_class[class_name] = accuracy

    global_accuracy = sum(class_correct) / sum(class_total) * 100
    return accuracies_per_class, global_accuracy

def plot_accuracies(global_accuracies, class_accuracies, epoch):
    global global_accuracy_plot, class_accuracy_plots

    if epoch == 0:
        plt.ion()  # Ativa o modo interativo
        plt.figure(figsize=(10, 5))

    # Precisão Global
    plt.subplot(1, 2, 1)
    if epoch == 0:
        global_accuracy_plot, = plt.plot(global_accuracies, label='Precisão Global')
    else:
        global_accuracy_plot.set_ydata(global_accuracies)
        global_accuracy_plot.set_xdata(range(1, epoch + 2))
    plt.xlabel('Época')
    plt.ylabel('Precisão (%)')
    plt.title('Precisão Global por Época')
    plt.legend()
    plt.xlim(1, epoch + 2)

    # Precisão por Classe
    plt.subplot(1, 2, 2)
    for class_name, accuracies in class_accuracies.items():
        if epoch == 0:
            class_accuracy_plots[class_name], = plt.plot(accuracies, label=class_name)
        else:
            class_accuracy_plots[class_name].set_ydata(accuracies)
            class_accuracy_plots[class_name].set_xdata(range(1, epoch + 2))
    plt.xlabel('Época')
    plt.ylabel('Precisão (%)')
    plt.title('Precisão por Classe por Época')
    plt.legend()
    plt.xlim(1, epoch + 2)

    plt.tight_layout()
    plt.draw()
    plt.pause(0.1)  # Pausa breve para permitir a renderização do gráfico
