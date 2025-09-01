from keras import Sequential, Input, Model, layers

import numpy as np

def player_embedder(full_play_dataset_preprocess):
    embedder = Sequential()
    embedder.add(layers.Embedding(input_dim= len(full_play_dataset_preprocess),output_dim=5, mask_zero=True))

    return embedder

def player_embedder_transform(X_to_embed):
    embedder = player_embedder(X_to_embed)
    embedder.compile("rmsprop","mse")
    X_transform = embedder.predict(np.array(X_to_embed))
    return X_transform
