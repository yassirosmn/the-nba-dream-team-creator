from keras import Sequential, Input, Model, layers
import numpy as np


def player_embedder_transform(X_to_embed):
    #initialize embedder
    embedder = Sequential()
    embedder.add(layers.Embedding(input_dim= len(X_to_embed),output_dim=7, mask_zero=True))

    #Embed
    embedder.compile("rmsprop","mse")
    X_transform = embedder.predict(X_to_embed)

    return X_transform


if __name__ == "__main__":
    from ml_logic.from_player_to_team import get_all_seasons_all_teams_starters_stats
    from ml_logic.registry import load_preprocessed_data_from_database

    X_preprocessed = load_preprocessed_data_from_database()
    temp1, temp2, X_to_embed = get_all_seasons_all_teams_starters_stats(X_preprocessed, False)
    X_transform = player_embedder_transform(X_to_embed)
    print(np.shape(X_transform))
    print("Test good (✅ pour Flavian)")
