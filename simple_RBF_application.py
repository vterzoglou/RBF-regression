from tensorflow import keras
from rbf_keras.rbflayer import *
from rbf_keras.kmeans_initializer import InitCentersKMeans
import numpy as np
from keras.datasets import boston_housing
from sklearn.preprocessing import minmax_scale
from keras import backend as K
import csv


def rsquare(y_true, y_pred):
    """
    Keras/tensor custom implementation for rsquare metric
    :param y_true: Tensor of predictions
    :param y_pred: Tensor of target values
    :return: Rsquare metric
    """
    SS_res = K.sum(K.square(y_true - y_pred))
    SS_tot = K.sum(K.square(y_true - K.mean(y_true)))
    rsq = 1-SS_res/(SS_tot + K.epsilon())
    if rsq >= -1:
        return rsq
    else:
        return float("NaN")


def preprocess_data(test_split=0.25):
    """
    Load, split and preprocess data
    :param test_split: float, percentage of total data to be used for testing
    :return: Train and test data tuples
    """
    (x_train, y_train), (x_test, y_test) = boston_housing.load_data(test_split=test_split)
    x_train = minmax_scale(x_train)
    y_train = minmax_scale(y_train)
    x_test = minmax_scale(x_test)
    y_test = minmax_scale(y_test)
    return (x_train, y_train), (x_test, y_test)


def create_model(feat_dim, rbf_neurons, x_train):
    """
    Creates a model consisting of the Fully connected cascade of:Input Layer -> RBF Layer -> Dense Intermediate Layer ->
    -> Dense output layer (neuron).
    The model is compiled with SGD of fixed Learning Rate, having MSE as its loss and RSquare as a watch metric
    :param feat_dim: The size of each input sample/vector
    :param rbf_neurons: The number of RBF neurons to be used
    :param x_train: The training data
    :return: The compiled model
    """
    model = keras.Sequential()
    model.add(keras.layers.InputLayer(input_shape=feat_dim))
    model.add(RBFLayer(rbf_neurons, initializer=InitCentersKMeans(x_train), trainable=False))
    model.add(keras.layers.Dense(128, activation='relu'))
    model.add(keras.layers.Dense(1, kernel_initializer=keras.initializers.Ones(), bias_initializer="zeros", trainable=False))
    model.compile(optimizer=keras.optimizers.SGD(learning_rate=0.001),
                  loss='mse', metrics=[rsquare])
    return model


if __name__ == "__main__":
    train_data, test_data = preprocess_data()
    x_train = train_data[0]
    train_samples, features_dim = x_train.shape
    percentages = [10, 50, 90]
    num_neurons = [int(np.round(percentage/1e2 * train_samples)) for percentage in percentages]
    models = [create_model(features_dim, neurons, x_train) for neurons in num_neurons]
    histories, stats = [], []
    for model in models:
        histories.append(model.fit(x_train, train_data[1], epochs=100, validation_split=0.2))
        stats.append(model.evaluate(test_data[0], test_data[1]))
    with open("simple_RBF_stats.csv", 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Percentage", "Number of Neurons", "Test RMSE", "Test RSquare"])
        for i, percentage in enumerate(percentages):
            writer.writerow([percentage, num_neurons[i], stats[i][0], stats[i][1]])
