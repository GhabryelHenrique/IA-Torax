import os
import pandas as pd
from PIL import Image
import gridfs
from pymongo import MongoClient
import io

def conectar_mongodb():
    client = MongoClient('mongodb://localhost:27017/')
    return client['admin'], gridfs.GridFS(client['admin'])

def salvar_imagem_no_mongodb(fs, image_path, metadata):
    with open(image_path, "rb") as image_file:
        im_bytes = image_file.read()
        image_id = fs.put(im_bytes, filename=os.path.basename(image_path), metadata=metadata)
    return image_id

def carregar_imagens(caminho_imagens, caminho_csv):
    db, fs = conectar_mongodb()
    df = pd.read_csv(caminho_csv)

    for index, row in df.iterrows():
        image_path = os.path.join(caminho_imagens, row['Image Index'])
        print(row)

        if os.path.exists(image_path):
            metadata = {
                "Finding Labels": row['Finding Labels'],
                "Follow-up #": row['Follow-up #'],
                "Patient ID": row['Patient ID'],
                "Patient Age": row['Patient Age'],
                "Patient Gender": row['Patient Gender'],
                "View Position": row['View Position'],
                "Original Dimensions Width": row['OriginalImage[Width'],
                "Original Dimensions Heigth": row['Height]'],
                "Pixel Spacing X": row['OriginalImagePixelSpacing[x'],
                "Pixel Spacing Y": row['y]']
            }
            image_id = salvar_imagem_no_mongodb(fs, image_path, metadata)
            print(f"Imagem {image_path} salva com ID: {image_id}")
        else:
            print(f"Imagem {image_path} não encontrada.")
