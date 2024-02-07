# T2_G1

Este repositório contém os códigos desenvolvidos no ambito do trabalho 2 de SAVI. O trabalho inclui um modelo de rede neural (`SimpleCNN`), um script principal (`classification.py`) para treino e classificação de imagens, um script de treino (`trainer.py`), e um módulo para criação do dataset personalizado (`dataset.py`).

## Estrutura do Código Fonte

### `classification.py`
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
- Reprodução do áudio descritivo da cena.

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

### `point_cloud_cluster.py`
Este módulo começa por criar uma nuvem de pontos através da combinação entre uma imagem rgb e uma imagem de profundidade da cena em questão. 
De seguida, procedeu-se à orientação do referencial para que este coincidi-se com o centro da mesa. Para tal, foi analisada a moda dos vetores normais que corresponde com o centro da mesa e a partir disso foi realizada a rotação do referencial. Para a translação do mesmo foi isolada a tampa da mesa, determinado o seu centróide e a partir disso determinou-se o vetor de translação.
Consequentemente, atraveś do centro do referencial construíu-se a bounding box, onde se isolou através de um Ramsac os objetos do tampo da mesa. Destes objetos isolados retirou-se algumas propriedades, nomeadamente a moda e a média da cor, a altura e a largura. Por fim gravou-se estas point cloud numa pasta.
Após isso, converteu-se a imagem rgb utilizada para hsv, devido à última ser preferivel aquando se aplica mascáras. Através das propriedades determinadas de cada objeto através da point cloud e através da realização de mascáras onde se usou comandos como o `erode` , o `dilation` e o `bitwase_and` , procurou-se na imagem o objeto com as propriedades equivalentes, sendo este recortado e guardado numa pasta para poder ser classificado.

## Como Utilizar

1. **Instalação de Dependências:**
   Certifique-se de ter as bibliotecas necessárias instaladas. Execute o seguinte comando para instalar as dependências:

   ```bash
   pip install torch torchvision scikit-learn tqdm

3. **Execução do Script Point_cloud_cluster:**
   Execute este script para obter as imagens. 
   Para usar este script terá que indicar no seu terminal a cena pretendida, como se encontra no exemplo seguinte. 
   Após escrever isto no seu terminal, o script irá começar a funcionar e irá obter as imagens recortadas e a sua representação na imagem rgb.

   ```bash 
   Ex: python3 point_cloud_cluster.py -sc (número da cena)
   
2. **Execução do Script Principal:**
   Execute o script principal classification.py para treinar ou classificar imagens. Modifique os parâmetros conforme necessário.
   ```bash
   ./classification.py
