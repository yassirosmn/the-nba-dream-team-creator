from tensorflow.keras import Sequential, Input, Model, layers

def player_embedder(get_starting_player_per_team_per_season : list):
    for player in get_starting_player_per_team_per_season:
        model = Sequential()
        model.add(Input(shape=player.shape[:1]))
        model.add(layers.Embedding(
        input_dim= player.shape[:1],
        output_dim= 50, # 100
        mask_zero=True, # Built-in masking layer :)
            ))
