# autoencoder model creation


"""imports"""


import random
import numpy as np
import cupy as cp
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from keras.datasets import mnist

import defs


"""data processing"""


# Load the MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# reshape the data
x_train = x_train.reshape((x_train.shape[0], -1)).astype('float32')
x_test = x_test.reshape((x_test.shape[0], -1)).astype('float32')
y_train = y_train.reshape((y_train.shape[0], -1)).astype('float32')
y_test = y_test.reshape((y_test.shape[0], -1)).astype('float32')

# normalize
x_train /= 255.
x_test /= 255.
y_train /= 255.
y_test /= 255.

# cut part of data
train_count = 60000
test_count = int(train_count * 0.2)
x_train, x_test, y_train, y_test = x_train[:train_count], x_test[:test_count], y_train[:train_count], y_test[:test_count]

# print shape of data
print(f"shape of data: {x_train.shape, y_train.shape, x_test.shape, y_test.shape}")


"""model creation and training"""


# list with layers hyperparams
nn_architecture = [
    {"input_dim": 784, "output_dim": 64, "activation": "relu"},
    {"input_dim": 64, "output_dim": 784, "activation": "sigmoid"},
]

# initialize model
model = defs.Model(nn_architecture)

# train model
params_values, cost_history, validation_cost_history, training_time = model.train(
    X=x_train,
    Y=x_train,
    validation_X=x_test,
    validation_Y=x_test,
    start_epoch=1,
    end_epoch=11,
    learning_rate=0.001,
    batch_size=256,
    path_to_params_checkpoints="checkpoints/",
    params_checkpoint_rate=50,
    loss_function=defs.mean_squared_error,
    )

# time spent on training
print(f"time spent on training: {training_time}")

# # train history

# figure size
plt.figure(figsize=(20, 10))

# add subplot
ax = plt.subplot(1, 2, 1)
# set title
ax.set_title(str("loss"))
# add image to subplot
ax.plot(cost_history)
ax.plot(validation_cost_history)
ax.legend(['train_loss', 'validation_loss'])

# add suptitle
plt.suptitle("history of model training")
# show plot
plt.show()


"""model test"""


# # load weights to model from jsonl
# model.load_params_from_jsonl(path_to_file="checkpoints/params_on_epoch_1000.jsonl")

# use model on test images
x_test_gen = model.full_forward_propagation(x_test, inference_mode=True)
print(f"x_test_gen.shape: {x_test_gen.shape}")
print(f"x_test.shape: {x_test.shape}")

# # show original and reconstructed samples

rows = 2  # number of rows in plot
columns = 20  # number of columns in plot
image_pass = 10  # number of skipped images between one added to subplot

# create figure
plt.figure(figsize=(columns, rows*2))
# elements counter
num_elements = 0
# iterate rows
for row in range(0, rows):
    # iterate columns and add elements
    for column in range(0, columns):
        # next element
        num_elements += 1

        # add subplot
        ax = plt.subplot(rows*2, columns, num_elements)
        # add original image to subplot
        ax.imshow(np.asarray(x_test).astype("float32").reshape([-1, 28, 28, 1])[num_elements * image_pass])
        # hide image shape
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # add subplot
        ax = plt.subplot(rows*2, columns, num_elements+rows*columns)
        # add generated image to subplot
        ax.imshow(np.asarray(x_test_gen).astype("float32").reshape([-1, 28, 28, 1])[num_elements * image_pass])
        # hide image shape
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

# add title
plt.suptitle("original and reconstructed data")

# show plots
plt.show()




