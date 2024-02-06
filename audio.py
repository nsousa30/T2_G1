#!/usr/bin/env python3
from gtts import gTTS
import pygame
import os

def convert_numbers_to_words(numbers):
    words = ['zero', 'um', 'dois', 'três', 'quatro', 'cinco', 'seis', 'sete', 'oito', 'nove', 'dez', 'onze', 'doze', 'treze', 'catorze']
    
    if len(numbers) == 1:
        return words[int(numbers)]
    elif len(numbers) == 2:
        if numbers.startswith('0'):
            return words[int(numbers[1])]
        elif numbers.startswith('1'):
            return words[int(numbers)]
        else:
            first_digit = words[int(numbers[0])]
            second_digit = words[int(numbers[1])] if numbers[1] != '0' else ''
            return f'{first_digit} {second_digit}'.strip()

output_point_dir = 'output_point_clouds'

# Verifica se há arquivos na pasta output_point_clouds
output_files = [f for f in os.listdir(output_point_dir) if os.path.isfile(os.path.join(output_point_dir, f))]

if output_files:
    # Obtém o caminho completo do primeiro arquivo na pasta output_point_clouds
    first_file = output_files[0]
    first_file_path = os.path.join(output_point_dir, first_file)
    
    # Extrai os dois primeiros números do nome do arquivo
    first_two_numbers = ''.join(filter(str.isdigit, first_file))[:2]

    # Converte os números em palavras
    numbers_in_words = convert_numbers_to_words(first_two_numbers)

    mytext = f'A sena {numbers_in_words} tem '

    language = 'pt'

    myobj = gTTS(text=mytext, lang=language, slow=False)
    myobj.save("scene.mp3")

    pygame.init()
    pygame.mixer.music.load("scene.mp3")
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():
        pygame.time.Clock().tick(10)

    # Apaga o arquivo de áudio
    os.remove("scene.mp3")

    # Apaga os arquivos na pasta output_point_clouds
    for file in output_files:
        file_path = os.path.join(output_point_dir, file)
        os.remove(file_path)

else:
    print("Não há arquivos na pasta output_point_clouds.")
