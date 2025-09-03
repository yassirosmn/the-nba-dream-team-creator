import numpy as np

from typing import Tuple

from tensorflow import keras
from keras import Model, Sequential, Input, layers, regularizers, optimizers
from keras.optimizers import Adam
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
    reg_l1 = regularizers.L1(0.02)
    reg_l2 = regularizers.L2(0.02)
    '''instanciate and return the CNN architecture of your choice with less than 150,000 params'''
    model = Sequential()
    model.add(Input(shape = (np.shape(X_train)[1],np.shape(X_train)[2]))) #
    ### First layer
    model.add(layers.Dense(512, activation='relu', kernel_regularizer=reg_l1))

    ### second layer
    model.add(layers.Dense(254,activation='relu'))

    ### third layer
    model.add(layers.Dense(126,activation='relu', kernel_regularizer=reg_l2))
    model.add(layers.Dropout(rate= 0.3))

    model.add(layers.Flatten()) #(40,)
    ### output layer
    model.add(layers.Dense(1, activation='relu'))

    return model

def initialize_deep_rnn_model(X_train):
    reg_l1 = regularizers.L1(0.01)
    '''instanciate and return the CNN architecture of your choice with less than 150,000 params'''
    ## RNN
    rnn = Sequential()
    rnn.add(Input(shape=(np.shape(X_train)[1],np.shape(X_train)[2]))),
    rnn.add(layers.LSTM(126,return_sequences=True, kernel_regularizer=reg_l1)),
    rnn.add(layers.LSTM(64,kernel_regularizer=reg_l1)),
    rnn.add(layers.Dense(30))
    rnn.add(layers.Dense(1, activation="relu"))
    return rnn

def initialize_deep_cnn_model(X_train):
    reg_l1 = regularizers.L1(0.01)
    '''instanciate and return the CNN architecture of your choice with less than 150,000 params'''
    # Conv1D
    cnn = Sequential()
    cnn.add(Input(shape=(np.shape(X_train)[1],np.shape(X_train)[2]))), #(np.shape(X_train)[1], np.shape(X_train)[2])
    cnn.add(layers.Conv1D(480, kernel_size=(1))),

    cnn.add(layers.Conv1D(240,kernel_regularizer=reg_l1, kernel_size=(1))),

    cnn.add(layers.Conv1D(120,kernel_regularizer=reg_l1, kernel_size=(1),)),

    cnn.add(layers.Conv1D(30, kernel_size=(1))),

    cnn.add(layers.Flatten())
    cnn.add(layers.Dense(30, activation= 'relu'))

    cnn.add(layers.Dense(1, activation="relu"))
    return cnn

def compile_deep_model(model,learning_rate = 0.001,epsilon = 1e-6):
    '''return a compiled model suited for the CIFAR-10 task'''
    pimp_my_optimizer = Adam(learning_rate=learning_rate,epsilon=epsilon)
    model.compile(
        loss = 'mae',
        optimizer = pimp_my_optimizer,
        metrics = ['mae']
    )
    return model

def fit_deep_model(model,X_train,y_train,validation_data, batch_size = 26, epochs= 500,verbose ='auto'):
    es = EarlyStopping(patience = 5, restore_best_weights=True)
    history = model.fit(X_train,
          y_train,
          validation_data= validation_data,
          batch_size = batch_size,
          epochs= epochs,
          verbose = verbose,
          callbacks = [es])
    return history, model


if __name__ == "__main__":



    print("Test good (✅ pour Flavian)")
