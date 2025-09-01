from keras import Sequential, Input, Model, layers
import numpy as np


def player_embedder_transform(X_to_embed):
    #initialize embedder
    embedder = Sequential()
    embedder.add(layers.Embedding(input_dim= len(X_to_embed),output_dim=5, mask_zero=True))

    #Embed
    embedder.compile("rmsprop","mse")
    X_transform = embedder.predict(np.array(X_to_embed))

    return X_transform
