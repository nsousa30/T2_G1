import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np

class RGBDDataset(Dataset):
    """
    Dataset personalizado para carregar imagens RGB-D.
    """
    def __init__(self, directory, transform=None):
        """
        Inicializa o dataset com o caminho do diretório e a transformação a ser aplicada.

        Args:
            directory (str): Caminho do diretório contendo as subpastas de categorias.
            transform (callable, optional): Transformação opcional a ser aplicada nas amostras.
        """
        self.directory = directory
        self.transform = transform
        # Cria um mapeamento de categorias para inteiros de forma ordenada
        categories = sorted(os.listdir(directory))
        print("\nCategories:")
        print(categories)
       
        self.category_to_int = {category: i for i, category in enumerate(categories)}

        self.samples = []
        # Itera sobre todas as categorias e arquivos no diretório
        for category in categories:
            category_path = os.path.join(directory, category)
            for file_name in os.listdir(category_path):
                if file_name.endswith('_crop.png'):  # Seleciona apenas imagens RGB
                    rgb_path = os.path.join(category_path, file_name)
                    mask_name = file_name.replace('_crop.png', '_maskcrop.png')
                    mask_path = os.path.join(category_path, mask_name)
                    if os.path.isfile(mask_path):  # Verifica se a máscara correspondente existe
                        self.samples.append((rgb_path, mask_path, category))

    def get_class_mapping(self):
        """
        Retorna o mapeamento de inteiros para nomes de categorias.

        Returns:
            dict: Dicionário com mapeamento de índices para nomes de categorias.
        """
        return {i: category for category, i in self.category_to_int.items()}

    def __len__(self):
        """
        Retorna o número total de amostras no dataset.
        """
        return len(self.samples)

    def __getitem__(self, idx):
        """
        Retorna uma amostra do dataset no índice fornecido.

        Args:
            idx (int): Índice da amostra a ser carregada.

        Returns:
            tuple: Contém a imagem segmentada e o tensor da categoria correspondente.
        """
        rgb_path, mask_path, category = self.samples[idx]
        # Carrega e converte a imagem RGB e a máscara correspondente
        rgb_image = Image.open(rgb_path).convert('RGB')
        mask_image = Image.open(mask_path).resize(rgb_image.size, Image.NEAREST)
        
        # Aplica as transformações definidas
        if self.transform:
            rgb_image = self.transform(rgb_image)
            mask_image = self.transform(mask_image)

        # Cria um tensor binário da máscara e aplica na imagem RGB
        mask_tensor = torch.tensor(np.array(mask_image), dtype=torch.uint8) > 0
        segmented_image = torch.tensor(np.array(rgb_image), dtype=torch.float32) * mask_tensor.float()

        # Converte o nome da categoria em um tensor
        category_tensor = torch.tensor(self.category_to_int[category], dtype=torch.long)

        return segmented_image, category_tensor
