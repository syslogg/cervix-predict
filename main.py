
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

# Link: https://keras.io/examples/conv_filter_visualization/
# Link: https://medium.com/nybles/create-your-first-image-recognition-classifier-using-cnn-keras-and-tensorflow-backend-6eaab98d14dd
# Link: https://www.pyimagesearch.com/2018/12/31/keras-conv2d-and-convolutional-layers/

# Leitura do banco de imagem
df = pd.read_csv('ds.csv')

# -=== Estudo sobre o Banco de Imagem Herlev ===-

# - Quantidade de imagens para cada valor:

df['diagnostic'].value_counts()

# 1. Montagem do Dataset
herlev_ds = []
for i, row in df.head(30).iterrows():
    img = np.array(Image.open(row[0]))
    herlev_ds = np.append(herlev_ds, [img, row[1]])

#herlev_ds = np.append(herlev_ds)

# 2. Montagem da Rede Neural Convolucional

def mount_neural_network():
    cnn = Sequential()
    cnn.add(Conv2D())