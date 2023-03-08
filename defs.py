# all classes and functions for model creation, but now on gpu


"""imports"""
import random
import time
from logger import logger
import json

import numpy as np
import cupy as cp
import pandas as pd

from matplotlib import pyplot as plt
import seaborn as sns


"""functions and classes"""


# mse loss function
def mean_squared_error(act, pred):
    """
    mean squared error loss function for loss calculation
    :param act: actual values
    :param pred: predicted values
    :return: calculated mse loss of data
    """
    diff = pred - act
    differences_squared = diff ** 2
    mean_diff = differences_squared.mean()
    return mean_diff


# binary accuracy function
def get_accuracy_value(Y_hat, Y):
    """
    binary accuracy calculating function
    :param Y_hat: generated probabilities
    :param Y: real classes
    :return: mean accuracy value between 0. and 1.
    """
    # transpose
    Y_hat, Y = cp.transpose(Y_hat), cp.transpose(Y)
    # convert probability into classes
    Y_hat_ = cp.asarray([1. if e > 0.5 else 0. for e in Y_hat], dtype="int32")
    # return mean of similarity between generated and real classes
    return cp.asarray([Y_hat_[e] == Y[e] for e in range(len(Y_hat_))], dtype="float32").mean()


# class for model object
class Model:

    # init function
    def __init__(self, nn_architecture):
        """
        model layers parameters initialization
        :param nn_architecture: dict with nn architecture
        """
        # initialize model params
        self.nn_architecture = nn_architecture
        self.params_values = self.init_layers()
        # initialize adam variables
        self.m, self.v = self.initialize_adam()

    # layers initialisation function
    def init_layers(self):
        """
        layers initialisation function
        :return: initialized model params
        """
        # number of layers
        number_of_layers = len(self.nn_architecture)
        # dict with all params and their values
        params_values = {}

        # iterate list of layers params
        for idx, layer in enumerate(self.nn_architecture):
            # take params of the layer
            layer_idx = idx + 1
            layer_icput_size = layer["input_dim"]
            layer_output_size = layer["output_dim"]

            # create weights and biases with random values for layer and add to params dict
            params_values['W' + str(layer_idx)] = cp.random.randn(
                layer_output_size, layer_icput_size) * 0.1
            params_values['b' + str(layer_idx)] = cp.random.randn(
                layer_output_size, 1) * 0.1

        # return dict with all neural network parameters
        return params_values

    # sigmoid function
    def sigmoid(self, Z):
        """
        sigmoid function
        """
        return 1/(1+cp.exp(-Z))

    # relu function
    def relu(self, Z):
        """
        relu function
        """
        return cp.maximum(0,Z)

    # backward sigmoid function
    def sigmoid_backward(self, dA, Z):
        """
        backward sigmoid function
        """
        sig = self.sigmoid(Z)
        return dA * sig * (1 - sig)

    # backward relu function
    def relu_backward(self, dA, Z):
        """
        backward relu function
        """
        dZ = cp.array(dA, copy=True)
        dZ[Z <= 0] = 0
        return dZ

    # function for raw input data preprocessing
    def input_data_processing(self, data):
        """
        function for raw input data preprocessing
        :param data: input data
        :return: transferred to device array and transposed data
        """
        return cp.transpose(cp.asarray(data, dtype="float32"))

    # function for output data preprocessing
    def output_data_processing(self, data):
        """
        function for preprocess data before output
        :param data: device array
        :return: transferred to host array and transposed data
        """
        return np.transpose(np.asarray(data.get(), dtype="float32"))

    # single layer forward propagation function
    def single_layer_forward_propagation(self, A_prev, W_curr, b_curr, activation="relu"):
        """
         single layer forward propagation function
        :param A_prev: input values
        :param W_curr: weights values
        :param b_curr: biases values
        :param activation: activation function name
        :return: layer output values after and before activation function
        """
        # multiply icput values to weights and add biases
        Z_curr = cp.dot(W_curr, A_prev) + b_curr

        # use one of activation function on current values
        if activation == "relu":
            activation_func = self.relu
        elif activation == "sigmoid":
            activation_func = self.sigmoid
        else:
            raise Exception('Non-supported activation function')

        # return current values with and without use of activation function
        return activation_func(Z_curr), Z_curr

    # full forward propagation function
    def full_forward_propagation(self, X, inference_mode=False):
        """
        full forward propagation function
        :param X: input data
        :param inference_mode: mode of use (in inference mode data been processed and only output data returned
        :return: output data and memory array with all intermediate stages of data
        """

        # preprocess data in inference mode:
        if inference_mode:
            X = self.input_data_processing(X)

        # dict with values from every layer
        memory = {}
        # icput values
        A_curr = X

        # iterate model layers
        for idx, layer in enumerate(self.nn_architecture):
            layer_idx = idx + 1
            A_prev = A_curr

            # use forward propagation on layer
            activ_function_curr = layer["activation"]
            W_curr = self.params_values["W" + str(layer_idx)]
            b_curr = self.params_values["b" + str(layer_idx)]
            A_curr, Z_curr = self.single_layer_forward_propagation(A_prev, W_curr, b_curr, activ_function_curr)

            # save previous and current values after weights and bias computation every layer
            memory["A" + str(idx)] = A_prev
            memory["Z" + str(layer_idx)] = Z_curr

        # return only processed output in inference mode
        if inference_mode:
            return self.output_data_processing(A_curr)
        else:
            # return current values and array with all intermediate states
            return A_curr, memory

    # single layer backward propagation function
    def single_layer_backward_propagation(self, dA_curr, W_curr, b_curr, Z_curr, A_prev, activation="relu"):
        """
        single layer backward propagation function
        :param dA_curr: derivation values
        :param W_curr: weights values
        :param b_curr: biases values
        :param Z_curr: current output values
        :param A_prev: previous input values
        :param activation: derivative activation function name
        :return: single layer backward propagation results
        """
        # form of previous input data?
        m = A_prev.shape[1]

        # initialize derivation function
        if activation == "relu":
            backward_activation_func = self.relu_backward
        elif activation == "sigmoid":
            backward_activation_func = self.sigmoid_backward
        else:
            raise Exception('Non-supported activation function')

        # calculate derivation of activation function in input values?
        dZ_curr = backward_activation_func(dA_curr, Z_curr)
        # calculate derivation of weights?
        dW_curr = cp.dot(dZ_curr, A_prev.T) / m
        # calculate derivation of biases?
        db_curr = cp.sum(dZ_curr, axis=1, keepdims=True) / m
        # calculate derivation of output data?
        dA_prev = cp.dot(W_curr.T, dZ_curr)
        # return derivations?
        return dA_prev, dW_curr, db_curr

    # full backward propagation function
    def full_backward_propagation(self, Y_hat, Y, memory):
        """
        full backward propagation function
        :param Y_hat: predicted output values
        :param Y: real output values
        :param memory: intermediate values between layers
        :return: calculated gradient values
        """
        # init gradients values dict
        grads_values = {}
        # reshape and form of input data?
        m = Y.shape[1]
        Y = Y.reshape(Y_hat.shape)

        # calculate previous input state?
        dA_prev = - (cp.divide(Y, Y_hat) - cp.divide(1 - Y, 1 - Y_hat))

        # iterate layers and their hyperparameters in backward sequence
        for layer_idx_prev, layer in reversed(list(enumerate(self.nn_architecture))):
            # layers hyperparams
            layer_idx_curr = layer_idx_prev + 1
            activ_function_curr = layer["activation"]
            # current input data?
            dA_curr = dA_prev

            # take target values from memory and params dicts
            A_prev = memory["A" + str(layer_idx_prev)]
            Z_curr = memory["Z" + str(layer_idx_curr)]
            W_curr = self.params_values["W" + str(layer_idx_curr)]
            b_curr = self.params_values["b" + str(layer_idx_curr)]

            # use them in single layer backward propagation function and take their derivations
            dA_prev, dW_curr, db_curr = self.single_layer_backward_propagation(
                dA_curr, W_curr, b_curr, Z_curr, A_prev, activ_function_curr)

            # add weights and biases derivations to gradient dict
            grads_values["dW" + str(layer_idx_curr)] = dW_curr
            grads_values["db" + str(layer_idx_curr)] = db_curr

        # return gradient of full backward propagation
        return grads_values

    # Adam optimizer parameters initialization
    def initialize_adam(self):
        """
        Adam optimizer parameters initialization
        :return: m and v vectors
        """
        m = {}  # first moment vector
        v = {}  # second moment vector

        # Initialize m, v. Input: "parameters". Outputs: "v, s".
        # iterate layers and hyperparams of them
        for layer_idx, layer in enumerate(self.nn_architecture):
            layer_idx_curr = layer_idx  # + 1

            # create vector arrays with zeros for all parameters in layers
            m["dW" + str(layer_idx_curr + 1)] = cp.zeros_like(self.params_values["W" + str(layer_idx_curr + 1)])
            m["db" + str(layer_idx_curr + 1)] = cp.zeros_like(self.params_values["b" + str(layer_idx_curr + 1)])
            v["dW" + str(layer_idx_curr + 1)] = cp.zeros_like(self.params_values["W" + str(layer_idx_curr + 1)])
            v["db" + str(layer_idx_curr + 1)] = cp.zeros_like(self.params_values["b" + str(layer_idx_curr + 1)])

        # return vectors
        return m, v

    # model parameters update using Adam optimizer
    def update_parameters_with_adam(self, grads_values, t=1, learning_rate=0.01,
                                    beta1=0.9, beta2=0.999, epsilon=1e-8):
        """
        model parameters update using Adam optimizer
        :param grads_values: gradient values
        :param t: t vector
        :param learning_rate: learning rate (Alpha) ratio
        :param beta1: momentum beta ratio
        :param beta2: RMSProp beta ratio
        :param epsilon: value what been added to square root of v vector
        :return: updated model parameters
        """

        m_corrected = {}  # Initializing first moment estimate, python dictionary
        v_corrected = {}  # Initializing second moment estimate, python dictionary

        # Perform Adam update on all parameters
        # iterate layers and hyperparams of them
        for layer_idx, layer in enumerate(self.nn_architecture):
            layer_idx_curr = layer_idx  # + 1

            # Moving average of the gradients. Inputs: "m, grads, beta1". Output: "m".
            self.m["dW" + str(layer_idx_curr + 1)] = beta1 * self.m["dW" + str(layer_idx_curr + 1)] + (1 - beta1) * grads_values["dW" + str(layer_idx_curr + 1)]
            self.m["db" + str(layer_idx_curr + 1)] = beta1 * self.m["db" + str(layer_idx_curr + 1)] + (1 - beta1) * grads_values["db" + str(layer_idx_curr + 1)]

            # Compute bias-corrected first moment estimate. Inputs: "m, beta1, t". Output: "m_corrected".
            m_corrected["dW" + str(layer_idx_curr + 1)] = self.m["dW" + str(layer_idx_curr + 1)] / (1 - cp.power(beta1, t))
            m_corrected["db" + str(layer_idx_curr + 1)] = self.m["db" + str(layer_idx_curr + 1)] / (1 - cp.power(beta1, t))

            # Moving average of the squared gradients. Inputs: "v, grads, beta2". Output: "v".
            self.v["dW" + str(layer_idx_curr + 1)] = beta2 * self.v["dW" + str(layer_idx_curr + 1)] + (1 - beta2) * cp.power(grads_values["dW" + str(layer_idx_curr + 1)], 2)
            self.v["db" + str(layer_idx_curr + 1)] = beta2 * self.v["db" + str(layer_idx_curr + 1)] + (1 - beta2) * cp.power(grads_values["db" + str(layer_idx_curr + 1)], 2)

            # Compute bias-corrected second raw moment estimate. Inputs: "v, beta2, t". Output: "v_corrected".
            v_corrected["dW" + str(layer_idx_curr + 1)] = self.v["dW" + str(layer_idx_curr + 1)] / (1 - cp.power(beta2, t))
            v_corrected["db" + str(layer_idx_curr + 1)] = self.v["db" + str(layer_idx_curr + 1)] / (1 - cp.power(beta2, t))

            # Update parameters. Icputs: "parameters, learning_rate, m_corrected, v_corrected, epsilon". Output: "parameters".
            self.params_values["W" + str(layer_idx_curr + 1)] = self.params_values["W" + str(layer_idx_curr + 1)] - learning_rate * m_corrected["dW" + str(layer_idx_curr + 1)] / (
                        cp.sqrt(v_corrected["dW" + str(layer_idx_curr + 1)]) + epsilon)
            self.params_values["b" + str(layer_idx_curr + 1)] = self.params_values["b" + str(layer_idx_curr + 1)] - learning_rate * m_corrected["db" + str(layer_idx_curr + 1)] / (
                        cp.sqrt(v_corrected["db" + str(layer_idx_curr + 1)]) + epsilon)

    # generator for splitting input data to bathes
    def batch_generator(self, x, y, batch_size=500):
        """
        generator for splitting input data into batches and return by pieces
        :param x: input data array
        :param y: output data array
        :param batch_size: size of one batch
        :return: data separated to n batches
        """
        # calculate number of full batches (everything  )
        step = x.shape[-1] // batch_size
        # iterate staps and yield one batch of data per time
        for count in range(step):
            # take next batch from full data and yield it
            data_batch_x = cp.transpose(cp.transpose(x)[count * batch_size:(count + 1) * batch_size])
            data_batch_y = cp.transpose(cp.transpose(y)[count * batch_size:(count + 1) * batch_size])
            yield data_batch_x, data_batch_y

    # function for saving weights and biases parameters to jsonl file in training process
    def save_params_to_jsonl(self, path, epoch_number):
        """
        function for saving weights and biases parameters to jsonl file
        :param path: path to directory where been writen all files
        :param epoch_number: number of current epoch
        :return: jsonl file with model parameters
        """
        # open new file
        with open(f"{path}/params_on_epoch_{epoch_number}.jsonl", "w", encoding="utf-8") as write_file:
            # convert parameters from arrays to lists and write to jsonl
            json.dump({key: self.params_values[key].get().tolist() for key in self.params_values}, write_file,
                      separators=(", ", ": "))
            # close file
            write_file.close()

    # function for loading weights and biases parameters from jsonl file
    def load_params_from_jsonl(self, path_to_file):
        """
        function for saving weights and biases parameters to jsonl file
        :param path_to_file: path to jsonl file with parameters
        :return: old model parameters replaced by loaded
        """
        # read json file
        with open(str(path_to_file), "r", encoding="utf-8") as read_file:
            # read json by lines and convert to list of dicts
            self.params_values = dict(json.loads(read_file.read()))
            # convert lists in dictionary to numpy arrays
            self.params_values = {key: cp.asarray(self.params_values[key], dtype="float32")
                                  for key in self.params_values}

    # training function
    def train(self, X, Y, validation_X, validation_Y, start_epoch, end_epoch, learning_rate, batch_size, path_to_params_checkpoints,
              params_checkpoint_rate, loss_function):
        """
        training function
        :param X: input train values
        :param Y: output train values
        :param validation_X: input validation values
        :param validation_Y: output validation values
        :param start_epoch: number of first epoch
        :param end_epoch: number of last epoch
        :param learning_rate: learning rate ration
        :param batch_size: size of one data batch
        :param path_to_params_checkpoints: path to directory where all checkpoints been saved
        :param params_checkpoint_rate: checkpoints been saved every params_checkpoint_rate epoch
        :param loss_function: loss calculation function
        :return: trained model parameters, training loss history, validation loss history, time spended on training
        """
        # lists for training loss history
        loss_history = []
        validation_loss_history = []

        # take start time of training
        start_time = time.time()

        # preprocess training data
        X, Y, validation_X, validation_Y = self.input_data_processing(X), self.input_data_processing(Y),\
            self.input_data_processing(validation_X), self.input_data_processing(validation_Y)

        # iterate for n epochs
        for epoch in range(start_epoch, end_epoch):
            # list for all loss values of one epoch
            loss_per_epoch = []
            validation_loss_per_epoch = []
            # create generators for data
            train_gen = self.batch_generator(X, Y, batch_size)
            validation_gen = self.batch_generator(validation_X, validation_Y, batch_size)

            # iterate train generator and optimize model params
            for x, y in train_gen:
                # make full forward propagation
                y_hat, cashe = self.full_forward_propagation(x)
                # calculate gradient
                grads_values = self.full_backward_propagation(y_hat, y, cashe)
                # optimize model params by gradient with Adam
                self.update_parameters_with_adam(grads_values, learning_rate=learning_rate)
                # calculate loss and add to loss list
                loss_per_epoch.append(loss_function(cp.transpose(y_hat), cp.transpose(y)))

            # iterate validation generator and calculate validation loss
            for x, y in validation_gen:
                # make full forward propagation
                y_hat, cashe = self.full_forward_propagation(x)
                # calculate loss and add to loss list
                validation_loss_per_epoch.append(loss_function(cp.transpose(y_hat), cp.transpose(y)))

            # save params to jsonl every params_checkpoint_rate epoch
            if epoch % params_checkpoint_rate == 0:
                self.save_params_to_jsonl(path=path_to_params_checkpoints, epoch_number=epoch)

            # calculate mean loss value of current epoch
            epoch_train_loss = cp.mean(cp.asarray(loss_per_epoch, dtype="float32"), axis=0, dtype="float32")
            epoch_validation_loss = cp.mean(cp.asarray(validation_loss_per_epoch, dtype="float32"), axis=0, dtype="float32")
            # add to loss history
            loss_history.append(epoch_train_loss)
            validation_loss_history.append(epoch_validation_loss)

            # inform about epoch completion
            logger.info(f"epoch {epoch} complete with train loss {epoch_train_loss} and validation loss {epoch_validation_loss}")

        # calculate time in seconds used for full training process
        training_time = time.time() - start_time

        # return trained model params, training loss and accuracy
        return self.params_values, cp.asarray(loss_history).get(), cp.asarray(validation_loss_history).get(), training_time






















