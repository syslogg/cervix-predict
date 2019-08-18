import numpy as np
import matplotlib.pyplot as plt
from keras.utils import to_categorical

def onehot_encode(data):
    encoded = to_categorical(data)
    return encoded

def onehot_decode(datum):
    return np.argmax(datum)
