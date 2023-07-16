from simple_RBF_application import preprocess_data
import os
from tensorflow import keras
from rbf_keras.rbflayer import *
from rbf_keras.kmeans_initializer import InitCentersKMeans
import numpy as np
import keras_tuner as kt
from sklearn.metrics import r2_score, mean_squared_error
from functools import partial

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def model_builder(hp, feat_dim, rbf_neurons, x_train):
    """
    Function to be used in tandem with partial method (from functools).
    Builds a hypermodel to search for optimal parameters
    :param hp: hyperparameters
    :param feat_dim: The size of each input sample/vector
    :param rbf_neurons: The number of RBF neurons to be used
    :param x_train: The training data
    :return: the hypermodel
    """
    model = keras.Sequential()

    # Input Layer
    model.add(keras.layers.InputLayer(input_shape=feat_dim))

    # RBF Layer
    percentages = [5, 15, 30, 50]
    hp_units1 = hp.Choice('RBF Layer units', values=rbf_neurons)
    model.add(RBFLayer(hp_units1, initializer=InitCentersKMeans(x_train), trainable=False))

    # Second Hidden Layer
    hp_units2 = hp.Choice('Second Layer units', values=[32, 64, 128, 256])
    model.add(keras.layers.Dense(hp_units2, activation='relu'))

    # Dropout Layer
    hp_drop = hp.Choice('Dropout probability', values=[0.2, 0.35, 0.5])
    model.add(keras.layers.Dropout(hp_drop, seed=42))

    # Output Layer
    model.add(
        keras.layers.Dense(1, kernel_initializer=keras.initializers.Ones(), bias_initializer="zeros", trainable=False))

    model.compile(optimizer=keras.optimizers.SGD(learning_rate=0.001), loss='mse')
    return model


if __name__ == "__main__":
    # Load, preprocess data, initialize variables to be used later
    train_data, test_data = preprocess_data()
    x_train, y_train = train_data
    train_samples, features_dim = x_train.shape
    percentages = [10, 50, 90]
    num_neurons = [int(np.round(percentage / 1e2 * train_samples)) for percentage in percentages]

    # Set up Hyperband parameter search
    tuner = kt.Hyperband(partial(model_builder,feat_dim=features_dim, rbf_neurons=num_neurons, x_train=x_train),
                         objective='val_loss', max_epochs=120, factor=2, hyperband_iterations=10, overwrite=True)
    tuner.search(x_train, y_train, epochs=120, validation_split=0.2)

    # Print best HPs and build final model
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    print(f"Best params:\n"
          f"RBF Neurons: {best_hps.get('RBF Layer units')}\n"
          f"Second hidden layer neurons: {best_hps.get('Second Layer units')}\n"
          f"Dropout prob: {best_hps.get('Dropout probability')}")
    final_model = tuner.hypermodel.build(best_hps)

    # Train and test final model
    history = final_model.fit(x_train, y_train, epochs=100, validation_split=0.2)
    y_preds = np.squeeze(final_model.predict(test_data[0]))
    y_test = test_data[1]
    test_r2 = r2_score(y_true=y_test, y_pred=y_preds)
    test_rmse = np.sqrt(mean_squared_error(y_true=y_test, y_pred=y_preds))
    print(f'Final model stats:\n'
          f'Test RSquare: {test_r2}\n'
          f'Test RMSE: {test_rmse}')
