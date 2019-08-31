# -*- coding: utf-8 -*-
"""
Created on Aug

@author: Hendy Rodrigues F.Silva
"""

import matplotlib.pyplot as plt
from keras.utils import to_categorical
import numpy as np
import pandas as pd
import os

#Data Augument
from keras.preprocessing.image import ImageDataGenerator
from PIL import Image
import utils
from network import NeuralNetwork
from imutils import paths
import cv2

from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split

#Metrics
from sklearn.metrics import classification_report

#Cross Validator
from sklearn.model_selection import KFold

BATCH_SIZE_HERLEV = 917

DIRECTORY_HERLEV = r'./dataset/herlev'

LABELS_HERLEV = set([
        'carcinoma_in_situ',
        'light_dysplastic',
        'moderate_dysplastic',
        'normal_columnar',
        'normal_intermediate',
        'normal_superficiel',
        'severe_dysplastic'])

# 1. Carregar Imagens

def load_images(path, labelsPath=[], target_resizer=(250, 250)):
    imagePaths = list(paths.list_images(path))
    data = []
    labels = []
    
    for imgPath in imagePaths:
        label = imgPath.split(os.path.sep)[-2]
        
        if label not in labelsPath:
            continue
        
        image = cv2.imread(imgPath)
        image = cv2.resize(image,target_resizer)
        
        data.append(image)
        labels.append(label)
        
    return np.array(data), labels
        

X, Y_without_encoded = load_images(DIRECTORY_HERLEV, LABELS_HERLEV)
X = X / 255

# Codificação One Hot
lb = LabelBinarizer()
Y = lb.fit_transform(Y_without_encoded)

# Divisão de DataSets de Treino e Teste
(X_train, X_test, Y_train, Y_test) = train_test_split(X, Y, test_size=0.25, stratify=Y, random_state=1)

# -=== Trabalhar com Data Augmentation ===-
dataGenerator = ImageDataGenerator(rotation_range=20, zoom_range=0.15,
                        	width_shift_range=0.2, height_shift_range=0.2, shear_range=0.15,
                        	horizontal_flip=True, fill_mode="nearest")


# 2. Montagem da Rede Neural Convolucional

model = NeuralNetwork.build(32,len(lb.classes_))
model.compile(
            loss='categorical_crossentropy', # Verificar cada parametro.
            optimizer='adam',
            metrics=['accuracy']
)

# 3. Treinamento da Rede Neural

STEPS_PER_EPOCH = len(X_train) 
EPOCHS = 5
BATCH_SIZE=32

result = model.fit_generator(dataGenerator.flow(X_train,Y_train, batch_size=BATCH_SIZE),
                             validation_data=(X_test, Y_test), steps_per_epoch=STEPS_PER_EPOCH, epochs=EPOCHS)

# 4. Predizer modelo de aprendizado

predictions = model.predict(X_test, batch_size=32)
print(classification_report(Y_test.argmax(axis=1),
	predictions.argmax(axis=1), target_names=lb.classes_))

# Grafico com Histórico de acuracia por Epoch

N = EPOCHS
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, N), result.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), result.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), result.history["acc"], label="train_acc")
plt.plot(np.arange(0, N), result.history["val_acc"], label="val_acc")
plt.title("Treinamento: Loss e Accuracy do Dataset HERLEV")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.show()


#plt.savefig(args["plot"])


# Leitura do banco de imagem

# -=== Estudo sobre o Banco de Imagem Herlev ===-



# - Quantidade de imagens para cada valor:

# 1. Montagem do Dataset


"""
train_datagen = ImageDataGenerator(
                    rescale=1./255,
                    shear_range=0.2,
                    zoom_range=0.2,
                    horizontal_flip=True,
                    validation_split=0.2)

valid_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train_generator = train_datagen.flow_from_directory(
    directory=r'./dataset/herlev',
    target_size=(250, 250),
    batch_size=32,
    color_mode='rgb',
    class_mode='categorical',
    seed=1,
    shuffle=True,
    subset='training'
)

validator_generator = valid_datagen.flow_from_directory(
    directory=r'./dataset/herlev',
    target_size=(250, 250),
    batch_size=32,
    color_mode='rgb',
    class_mode='categorical',
    seed=1,
    shuffle=True,
    subset='validation'
)
"""





# 3. Treinamento Rede Neural

"""
model.fit_generator(
        train_generator,
        steps_per_epoch=len(train_generator),
        epochs=10,
        validation_data=validator_generator,
        validation_steps=len(validator_generator)
)
"""

# 4. Prever Dados

# model.predict()

# 5. Postar Resultados



































