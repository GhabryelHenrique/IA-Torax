import numpy as np
import gridfs
from pymongo import MongoClient
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from PIL import Image
import io

def conectar_mongodb():
    client = MongoClient('mongodb://localhost:27017/')
    db = client['minha_database']
    fs = gridfs.GridFS(db)
    return fs

def carregar_imagens_do_mongodb(fs):
    imagens = []
    for grid_out in fs.find():
        im_bytes = grid_out.read()
        im = Image.open(io.BytesIO(im_bytes))
        im = im.resize((128, 128))  # Redimensiona a imagem para o tamanho esperado pelo modelo
        imagens.append(np.array(im))
    imagens = np.array(imagens) / 255.0  # Normaliza as imagens
    return imagens

def construir_modelo():
    modelo = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
        MaxPooling2D(2, 2),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(1, activation='sigmoid')  # Assumindo uma classificação binária
    ])
    modelo.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return modelo

def treinar_modelo(modelo, imagens, labels):
    modelo.fit(imagens, labels, epochs=10)  # Treinar o modelo
    return modelo

def processar_imagens():
    fs = conectar_mongodb()
    imagens = carregar_imagens_do_mongodb(fs)
    # Aqui, você precisaria ter os labels para treinar, isso é só um exemplo
    labels = np.random.randint(0, 2, imagens.shape[0])
    modelo = construir_modelo()
    modelo_treinado = treinar_modelo(modelo, imagens, labels)
    return modelo_treinado

if __name__ == '__main__':
    modelo_treinado = processar_imagens()
    modelo_treinado.save('modelo_treinado.h5')
