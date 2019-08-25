
import matplotlib.pyplot as plt
from keras.utils import to_categorical
from keras.models import Sequential
# Layers
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Conv2D
from keras.layers import MaxPool2D

#Data Augument
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import pandas as pd
from PIL import Image
import utils
from network import NeuralNetwork

BATCH_SIZE_HERLEV = 917

# Leitura do banco de imagem

# -=== Estudo sobre o Banco de Imagem Herlev ===-

# -=== Trabalhar com Data Augmentation ===-

train_datagen = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)

train_datagen.flow_from_directory(
    directory=r'./dataset/herlev',
    target_size=(250, 250),
    batch_size=32,
    color_mode='rgb',
    class_mode='categorical',
    seed=1,
    shuffle=True
)

# - Quantidade de imagens para cada valor:

# 1. Montagem do Dataset



# 2. Montagem da Rede Neural Convolucional

model = NeuralNetwork.build(32,10)
model.compile(loss='categorical_crossentropy', optimizer='sgd')