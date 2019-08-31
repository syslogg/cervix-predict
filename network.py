from keras.utils import to_categorical
from keras.models import Sequential
#import SpatialPyramidPooling

# Layers
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Conv2D
from keras.layers import MaxPool2D, Activation

class NeuralNetwork:
    @staticmethod
    def build(num_filter, num_class):
        cnn = Sequential()

        # Passo 1: Criar Camada Convolucional
        cnn.add(Conv2D(num_filter, kernel_size=(3, 3), activation='relu',input_shape=(250, 250, 3)))

        # Passo 2: Efetuar o MaxPooling
        cnn.add(MaxPool2D())
        
        num_filter = num_filter * 2
        cnn.add(Conv2D(num_filter, kernel_size=(3, 3), activation='relu',input_shape=(250, 250, 3)))
        cnn.add(MaxPool2D())
        
        # Passo 3: Flatten as saidas da camada convolucional
        cnn.add(Flatten())

        # Passo 4: Exibir saida com uma camada totalmente conectada (Fully-Connected)
        cnn.add(Dense(num_class))
        cnn.add(Activation('softmax'))
        #cnn.add(SpatialPyramidPooling([1, 2, 4])) # Pesquisar sobre

        return cnn