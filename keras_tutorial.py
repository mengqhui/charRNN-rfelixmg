import numpy as np
import timeit
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Activation, LSTM

class kerasTutorial():

    def __init__(self):
        self.model = Sequential()
        self.model.add(LSTM(32, input_shape=(25,64)))

    def run(self, data, labels):
        self.model.fit(data, labels, nb_epoch=50, batch_size=32)


if __name__ == "__main__":

    classifier = kerasTutorial()

    data = np.random.random((10000, 784))
    labels = np.random.randint(2, size=(10000, 1))

    # train the model, iterating on the data in batches
    # of 32 samples
    classifier.run(data, labels)