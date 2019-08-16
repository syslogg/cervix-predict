import utils as u
from keras.datasets import mnist
import matplotlib.pyplot as plt
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten
import numpy as np

#download mnist data and split into train and test sets
(X_train, y_train), (X_test, y_test) = mnist.load_data()
