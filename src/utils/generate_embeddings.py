from pandas import read_csv
from keras.layers import Input, Embedding, Concatenate, Dense  # type: ignore
from keras.models import Model, save_model  # type: ignore
from keras.callbacks import EarlyStopping, ModelCheckpoint  # type: ignore
from keras.optimizers import Adam  # type: ignore
import numpy as np
from math import floor
import pdb
import pickle


from typing import Dict, Any

if __name__ == '__main__':
    # Get Travel Times
    travel_times = read_csv('../../data/ny/zone_traveltime.csv', header=None).values
    mean_val = np.mean(travel_times)
    max_val = np.abs(travel_times).max()
    print("Mean: {}, Max: {}".format(mean_val, max_val))
    travel_times -= mean_val
    travel_times /= max_val

    # Define NN
    origin_input = Input(shape=(1,), name='origin_input')
    destination_input = Input(shape=(1,), name='destination_input')

    location_embed = Embedding(output_dim=10, input_dim=travel_times.shape[0] + 1, mask_zero=True, name='location_embedding')
    origin_embed = location_embed(origin_input)
    destination_embed = location_embed(destination_input)

    state_embed = Concatenate()([origin_embed, destination_embed])
    state_embed = Dense(100, activation='elu', name='state_embed_1')(state_embed)
    output = Dense(1, name='output')(state_embed)

    model = Model(inputs=[origin_input, destination_input], outputs=output)
    model.compile(optimizer=Adam(), loss='mean_squared_error')

    # Format
    X: Dict[str, Any] = {'origin_input': [], 'destination_input': []}
    y = []
    for origin in range(travel_times.shape[0]):
        for destination in range(travel_times.shape[1]):
            X['origin_input'].append(origin + 1)
            X['destination_input'].append(destination + 1)
            y.append(travel_times[origin, destination])

    # Get train/test split
    idxs = np.array(list(range(len(y))))
    np.random.shuffle(idxs)

    train_idxs = idxs[0:floor(0.8 * len(y))]
    valid_idxs = idxs[floor(0.8 * len(y)) + 1:floor(0.9 * len(y))]
    test_idxs = idxs[floor(0.9 * len(y)) + 1:]

    X_train = {key: np.array(value)[train_idxs] for key, value in X.items()}
    X_valid = {key: np.array(value)[valid_idxs] for key, value in X.items()}
    X_test = {key: np.array(value)[test_idxs] for key, value in X.items()}
    y_train = (np.array(y)[train_idxs]).reshape((-1, 1, 1))
    y_valid = (np.array(y)[valid_idxs]).reshape((-1, 1, 1))
    y_test = (np.array(y)[test_idxs]).reshape((-1, 1, 1))

    # Train
    model.fit(x=X_train, y=y_train, validation_data=(X_valid, y_valid), batch_size=1024, epochs=1000, callbacks=[EarlyStopping(patience=15), ModelCheckpoint('../../models/embedding.h5', save_best_only=True)])
    test_loss = model.evaluate(x=X_test, y=y_test)
    print("Loss on test fraction: {}".format(test_loss))

    # Save Embeddings
    pickle.dump(model.layers[2].get_weights(), open('../../data/ny/embedding_weights.pkl', 'wb'))
