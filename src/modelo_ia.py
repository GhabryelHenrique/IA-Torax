import numpy as np
import gridfs
from pymongo import MongoClient
from PIL import Image
import io
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import pandas as pd
from dotenv import load_dotenv
import os

def conectar_mongodb():
    print('Conectando ao mongo')
    client = MongoClient('mongodb://localhost:27017/')
    db = client['admin']
    fs = gridfs.GridFS(db)
    return db, fs

def carregar_labels_do_csv(caminho_csv):
    df = pd.read_csv(caminho_csv)
    labels_dict = {}
    for index, row in df.iterrows():
        image_id = row['id']
        labels = row[1:-1].values.astype(int) 
        labels_dict[image_id] = labels
    return labels_dict

def carregar_lista_imagens(caminho_txt):
    with open(caminho_txt, 'r') as file:
        lista_imagens = file.read().splitlines()
    return lista_imagens

def carregar_imagens_do_mongodb(db, fs, labels_dict, lista_imagens):
    imagens = []
    labels = []
    
    query = {"filename": {"$in": list(lista_imagens)}}
    total_imagens = db.fs.files.count_documents(query)
    imagens_processadas = 0

    for grid_out in fs.find(query):
        imagens_processadas += 1
        
        im_bytes = grid_out.read()
        im = Image.open(io.BytesIO(im_bytes))
        im = im.convert('RGB')
        im = im.resize((128, 128))
        imagens.append(np.array(im))
        image_name = grid_out.filename

        if image_name in labels_dict:
            labels.append(labels_dict[image_name])
        else:
            labels.append(np.zeros(20))

        print(f'Imagens processadas: {imagens_processadas}, Imagens restantes: {total_imagens - imagens_processadas}')

    return np.array(imagens), np.array(labels)

def construir_modelo():
    print('Construindo Modelo')
    modelo = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
        MaxPooling2D(2, 2),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(20, activation='softmax')  # Ajuste aqui para 20 classes
    ])
    modelo.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])  # Ajuste aqui para categorical_crossentropy
    return modelo

def treinar_modelo(modelo, imagens, labels):
    print('Treinando Modelo')
    if imagens.size == 0 or labels.size == 0:
        raise ValueError("Imagens ou labels estão vazios. Verifique se os dados foram carregados corretamente.")
    
    datagen = ImageDataGenerator(rescale=1.0/255.0)
    train_generator = datagen.flow(imagens, labels, batch_size=32)
    modelo.fit(train_generator, epochs=10)
    return modelo

def processar_imagens(caminho_csv, caminho_txt):
    print('Processando Imagens')
    db, fs = conectar_mongodb()
    labels_dict = carregar_labels_do_csv(caminho_csv)
    lista_imagens = carregar_lista_imagens(caminho_txt)
    imagens, labels = carregar_imagens_do_mongodb(db, fs, labels_dict, lista_imagens)
    
    if imagens.size == 0 or labels.size == 0:
        raise ValueError("Nenhuma imagem ou label foi carregada. Verifique os caminhos e os dados fornecidos.")
    
    modelo = construir_modelo()
    modelo_treinado = treinar_modelo(modelo, imagens, labels)
    return modelo_treinado

if __name__ == '__main__':
    load_dotenv()

    caminho_csv = os.getenv('CAMINHO_CSV_LABELS')
    caminho_txt = os.getenv('CAMINHO_TXT')
    modelo_treinado = processar_imagens(caminho_csv, caminho_txt)
    modelo_treinado.save('modelo_treinado.h5')