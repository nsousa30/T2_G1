
import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np



class RGBDDataset(Dataset):
    def __init__(self, directory, transform=None):
        self.directory = directory
        self.transform = transform
        self.category_to_int = {category: i for i, category in enumerate(os.listdir(directory))}
        self.samples = []
        for category in os.listdir(directory):
            category_path = os.path.join(directory, category)
            for object_instance in os.listdir(category_path):
                instance_path = os.path.join(category_path, object_instance)
                for file_name in os.listdir(instance_path):
                    if file_name.endswith('_crop.png'):  # Imagem RGB
                        rgb_path = os.path.join(instance_path, file_name)
                        mask_name = file_name.replace('_crop.png', '_maskcrop.png')
                        mask_path = os.path.join(instance_path, mask_name)
                        # Verificar se a máscara existe antes de adicionar aos samples
                        if os.path.isfile(mask_path):
                            self.samples.append((rgb_path, mask_path, category))
                        else:
                            print(f"Arquivo de máscara ausente: {mask_path}")
                            

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        rgb_path, mask_path, category = self.samples[idx]
        # Carregar a imagem RGB e a máscara
        rgb_image = Image.open(rgb_path).convert('RGB')
        mask_image = Image.open(mask_path)
        mask_image = mask_image.resize(rgb_image.size, Image.NEAREST)  # Certifique-se de que a máscara é do mesmo tamanho que a imagem
        
        # Aplicar transformações (se houver)
        if self.transform is not None:
            rgb_image = self.transform(rgb_image)
            mask_image = self.transform(mask_image)

        # Transformar a máscara em um tensor binário
        mask_tensor = torch.tensor(np.array(mask_image), dtype=torch.uint8)
        mask_tensor = mask_tensor > 0  # Supondo que a máscara é binária

        # Criar uma versão segmentada da imagem
        rgb_tensor = torch.tensor(np.array(rgb_image), dtype=torch.float32)
        segmented_image = rgb_tensor * mask_tensor.float()
        
        # Codificar a categoria como um tensor
        category_tensor = torch.tensor(self.category_to_int[category])

        return segmented_image, category_tensor
