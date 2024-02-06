# T2_G1

Este repositório contém os códigos desenvolvidos no ambito do trabalho 2 de SAVI. O trabalho inclui um modelo de rede neural simples (`SimpleCNN`), um script principal (`main.py`) para treino e classificação de imagens, um script de treino (`trainer.py`), e um módulo para criação do dataset personalizado (`dataset.py`).

## Estrutura do Código Fonte

### `main.py`
O script principal é utilizado para treino e classificação de imagens. Ele contém as seguintes funcionalidades:

- Carrega as bibliotecas necessárias, como PyTorch, torchvision, scikit-learn, etc.
- Define a função `classify_loaded_image` para classificar uma imagem previamente carregada.
- Inicia o dispositivo de treino (CPU ou GPU).
- Define caminhos para o dataset, checkpoints e parâmetros de treino.
- Carrega o dataset, divide em conjuntos de treino e teste, e configura os DataLoaders.
- Cria o modelo, define o critério e o otimizador.
- Carrega um checkpoint pré-treinado, se disponível.
- Obtém o mapeamento de classes.
- Inicia o treino ou teste do modelo.

### `trainer.py`
O script de treino contém funções para treinar o modelo, guardar e carregar checkpoints, e monitorizar a melhoria na perda para aplicar early stopping. As principais funções incluem:

- `save_checkpoint`: Salva o estado atual do treino em um arquivo de checkpoint.
- `load_checkpoint`: Carrega um checkpoint se disponível, incluindo modelo, otimizador, perda e precisão.
- `train`: Função principal para treinar o modelo com suporte para early stopping.

### `dataset.py`
Este módulo define um dataset personalizado para carregar imagens RGB-D. A classe `RGBDDataset` inclui funcionalidades como:

- Inicialização com caminho do diretório e transformação opcional.
- Criação de um mapeamento de categorias para inteiros de forma ordenada.
- Carregamento das amostras do dataset, incluindo imagens RGB, máscaras correspondentes e categorias.
- Métodos para obter o mapeamento de classes, o número total de amostras e uma amostra específica.

### `model.py`
Este módulo contém a definição do modelo de rede neural `SimpleCNN`. A arquitetura é composta por duas camadas de convolução seguidas por camadas totalmente conectadas. Principais características:

- Camadas convolucionais para extração de características.
- Camadas totalmente conectadas para classificação.
- Método `_forward_conv` para definir a forma da camada de entrada para as camadas totalmente conectadas.
- Método `forward` para a passagem direta da rede.

## Como Utilizar

1. **Instalação de Dependências:**
   Certifique-se de ter as bibliotecas necessárias instaladas. Execute o seguinte comando para instalar as dependências:

   ```bash
   pip install torch torchvision scikit-learn tqdm

2. **Execução do Script Principal:**
   Execute o script principal main.py para treinar ou classificar imagens. Modifique os parâmetros conforme necessário.
   ```bash
   ./main.py
