#!/usr/bin/env python3

import os
import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt

def ply_to_rgb_image(ply_file_path, output_image_path):
    # Carregar o arquivo PLY
    point_cloud = o3d.io.read_point_cloud(ply_file_path)

    # Verificar se há cores no ponto de nuvem
    if not point_cloud.has_colors():
        print(f"O arquivo {ply_file_path} não contém informações de cores. Pulando.")
        return

    # Obter as coordenadas XYZ e as cores RGB
    xyz = np.asarray(point_cloud.points)
    rgb = np.asarray(point_cloud.colors) * 255  # Open3D armazena cores como valores float entre 0 e 1, convertendo para int no intervalo [0, 255]

    # Verificar se há pontos na nuvem
    if xyz.shape[0] == 0:
        print(f"O arquivo {ply_file_path} não contém pontos na nuvem. Pulando.")
        return

    # Normalizar as coordenadas para o intervalo [0, 1]
    normalized_xyz = (xyz - np.min(xyz, axis=0)) / (np.max(xyz, axis=0) - np.min(xyz, axis=0))

    # Criar uma imagem RGB
    image = np.zeros((xyz.shape[0], 3), dtype=np.uint8)
    image[:, :] = rgb

    # Redimensionar para uma imagem 2D
    image_width, image_height = 512, 512  # Defina o tamanho da imagem desejado
    reshaped_image = np.zeros((image_height, image_width, 3), dtype=np.uint8)

    # Mapear as coordenadas normalizadas para os índices da imagem
    pixel_coords = (normalized_xyz[:, :2] * [image_width - 1, image_height - 1]).astype(int)
    reshaped_image[pixel_coords[:, 1], pixel_coords[:, 0]] = image

    # Salvar a imagem
    plt.imsave(output_image_path, reshaped_image)

# Diretório contendo os arquivos PLY
ply_directory = 'output_point_clouds'

# Diretório de saída para as imagens RGB
output_image_directory = 'output_rgb_images'

# Garantir que o diretório de saída existe
os.makedirs(output_image_directory, exist_ok=True)

# Loop através de todos os arquivos PLY na pasta
for filename in os.listdir(ply_directory):
    if filename.endswith(".ply"):
        ply_file_path = os.path.join(ply_directory, filename)
        
        # Nome do arquivo de saída
        output_image_path = os.path.join(output_image_directory, os.path.splitext(filename)[0] + ".png")

        # Converter PLY para imagem RGB
        ply_to_rgb_image(ply_file_path, output_image_path)

         # Exibir a imagem
        img = plt.imread(output_image_path)
        plt.imshow(img)
        plt.title(f"Imagem gerada a partir de {filename}")
        plt.show()

print("Conversão concluída.")
