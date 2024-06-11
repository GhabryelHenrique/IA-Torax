import numpy as np
import gridfs
import os
import io
import pandas as pd
import joblib

from pymongo import MongoClient
from PIL import Image
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from dotenv import load_dotenv
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.metrics import Accuracy, Precision, Recall, AUC

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
        labels = row[1:].values.astype(int)  # Ajuste aqui para incluir todas as colunas exceto a primeira
        labels_dict[image_id] = labels
    return labels_dict

def carregar_lista_imagens(caminho_txt):
    with open(caminho_txt, 'r') as file:
        lista_imagens = file.read().splitlines()
    return lista_imagens

def calcular_porcentagem(parte, total):
    if total == 0:
        return 0
    return (parte / total) * 100

def carregar_imagens_do_mongodb(db, fs, labels_dict, lista_imagens):
    imagens = []
    labels = []
    
    query = {"filename": {"$in": list(lista_imagens)}}
    total_imagens = db.fs.files.count_documents(query)
    imagens_processadas = 0
    imagens_falhas = 0
    images =  fs.find(query)

    for grid_out in images:
        try:

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

            porcentagem = calcular_porcentagem(imagens_processadas, total_imagens)
            print(f'{porcentagem:.2f}% - Imagens processadas: {imagens_processadas}, Imagens restantes: {total_imagens - imagens_processadas}, Imagens não processadas: {imagens_falhas}')
        except:
            imagens_falhas += 1
            print(f'{porcentagem:.2f}% - Imagens processadas: {imagens_processadas}, Imagens restantes: {total_imagens - imagens_processadas}, Imagens não processadas: {imagens_falhas}')
    
    return np.array(imagens), np.array(labels)

def calcular_metricas(y_true, y_pred):
    report = classification_report(y_true, y_pred, output_dict=True)
    accuracy = accuracy_score(y_true, y_pred)
    
    return {
        'accuracy': accuracy,
        'classification_report': report
    }

def construir_modelo_cnn(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy']):
    print('Construindo Modelo')
    modelo = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
        MaxPooling2D(2, 2),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(20, activation='softmax')
    ])
    
    modelo.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    return modelo

def construir_modelo_knn():
    print('Construindo Modelo KNN')
    modelo = KNeighborsClassifier(n_neighbors=5)
    return modelo

def construir_modelo_random_forest():
    print('Construindo Modelo Random Forest')
    modelo = RandomForestClassifier(n_estimators=100, random_state=42)
    return modelo

def treinar_modelo_knn(modelo, imagens, labels):
    print('Treinando Modelo KNN')
    imagens_flat = imagens.reshape(imagens.shape[0], -1)
    modelo.fit(imagens_flat, labels)
    y_pred = modelo.predict(imagens_flat)
    metricas = calcular_metricas(labels, y_pred)
    return modelo, metricas

def treinar_modelo_random_forest(modelo, imagens, labels):
    print('Treinando Modelo Random Forest')
    imagens_flat = imagens.reshape(imagens.shape[0], -1)
    modelo.fit(imagens_flat, labels)
    y_pred = modelo.predict(imagens_flat)
    metricas = calcular_metricas(labels, y_pred)
    return modelo, metricas

def treinar_modelo_cnn(modelo, imagens, labels):
    print('Treinando Modelo')
    if imagens.size == 0 or labels.size == 0:
        raise ValueError("Imagens ou labels estão vazios. Verifique se os dados foram carregados corretamente.")
    
    datagen = ImageDataGenerator(
        rescale=1.0/255.0,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    
    train_generator = datagen.flow(imagens, labels, batch_size=32)
    modelo.fit(train_generator, epochs=100)
    return modelo

def processar_imagens(caminho_csv, caminho_txt):
    print('Processando Imagens')
    db, fs = conectar_mongodb()
    labels_dict = carregar_labels_do_csv(caminho_csv)
    lista_imagens = carregar_lista_imagens(caminho_txt)
    imagens, labels = carregar_imagens_do_mongodb(db, fs, labels_dict, lista_imagens)

    if imagens.size == 0 or labels.size == 0:
        raise ValueError("Nenhuma imagem ou label foi carregada. Verifique os caminhos e os dados fornecidos.")
    
    optimizer = Adam(learning_rate=0.0001)
    loss = CategoricalCrossentropy()
    metrics = [Accuracy(), Precision(), Recall(), AUC()]
    modelo_cnn = construir_modelo_cnn(optimizer=optimizer, loss=loss, metrics=metrics)
    modelo_cnn_treinado = treinar_modelo_cnn(modelo_cnn, imagens, labels)
    
    modelo_knn = construir_modelo_knn()
    modelo_knn_treinado, metricas_knn = treinar_modelo_knn(modelo_knn, imagens, labels)
    print("Métricas KNN:", metricas_knn)
    
    modelo_rf = construir_modelo_random_forest()
    modelo_rf_treinado, metricas_rf = treinar_modelo_random_forest(modelo_rf, imagens, labels)
    print("Métricas Random Forest:", metricas_rf)
    
    return modelo_cnn_treinado, modelo_knn_treinado, modelo_rf_treinado

if __name__ == '__main__':
    load_dotenv()
    
    caminho_csv = os.getenv('CAMINHO_CSV_LABELS')
    caminho_txt = os.getenv('CAMINHO_TXT')

    modelo_cnn_treinado, modelo_knn_treinado, modelo_rf_treinado = processar_imagens(caminho_csv, caminho_txt)
    
    # Salvar modelos
    modelo_cnn_treinado.save('modelo_cnn_treinado.h5')
    joblib.dump(modelo_knn_treinado, 'modelo_knn_treinado.pkl')
    joblib.dump(modelo_rf_treinado, 'modelo_rf_treinado.pkl')