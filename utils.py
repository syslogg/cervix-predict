import numpy as np
from keras.utils import to_categorical

def onehot_encode(data):
    encoded = to_categorical(data)
    return encoded

def onehot_decode(datum):
    return np.argmax(datum)

def hendy():
    return 'Testeeeee'