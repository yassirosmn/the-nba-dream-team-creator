import numpy as np

from typing import Tuple

from tensorflow import keras
from keras import Model, Sequential, Input, layers, regularizers, optimizers
from keras.callbacks import EarlyStopping


def initialize_model(model_type, input_shape : tuple = None) -> Model:
    """
    Initialize the model
    """

    model = model_type

    return model


def fit_model(model, X, y) :
    model.fit(X,y)
    return model



def score_model(model, X, y) :
    return model.score(X,y)




##########################
######### DEEP ###########
##########################

def initialize_deep_dense_model(X_train):
    #reg_l1 = regularizers.L1(0.01)
    '''instanciate and return the CNN architecture of your choice with less than 150,000 params'''
    model = Sequential()
    model.add(Input(shape = (40, 7)))

    ### First layer
    model.add(layers.Dense(300, activation='relu'))

    ### second layer
    model.add(layers.Dense(200,activation='relu'))

    ### third layer
    model.add(layers.Dense(126,activation='relu'))
    model.add(layers.Dropout(rate= 0.2))

    ### output layer
    model.add(layers.Dense(1, activation='relu'))

    return model

def initialize_deep_rnn_model(X_train):
    #reg_l1 = regularizers.L1(0.01)
    '''instanciate and return the CNN architecture of your choice with less than 150,000 params'''
    ## RNN
    rnn = Sequential()
    rnn.add(Input(shape=(X_train.shape[1],X_train.shape[2],1))),
    rnn.add(layers.LSTM(50)),

    rnn.add(layers.Dense(1, activation="relu"))
    return rnn

def initialize_deep_cnn_model(X_train):
    #reg_l1 = regularizers.L1(0.01)
    '''instanciate and return the CNN architecture of your choice with less than 150,000 params'''
    # Conv1D
    cnn = Sequential()
    cnn.add(Input(shape=(X_train.shape[1],X_train.shape[2],1))), #X_train.shape[1:]
    cnn.add(layers.Conv1D(20, kernel_size=(1))),
    cnn.add(layers.Dense(1, activation="relu"))
    return cnn

def compile_deep_model(model):
    '''return a compiled model suited for the CIFAR-10 task'''
    model.compile(
        loss = 'mae',
        optimizer = 'adam',
        metrics = ['mae']
    )
    return model

def fit_deep_model(model,X_train,y_train,validation_data, batch_size = 32, epochs= 200):
    es = EarlyStopping(patience = 20, restore_best_weights=True)
    history = model.fit(X_train,
          y_train,
          validation_data= validation_data,
          batch_size = batch_size,
          epochs= epochs,
          callbacks = [es])
    return history, model


if __name__ == "__main__":



    print("Test good (✅ pour Flavian)")
