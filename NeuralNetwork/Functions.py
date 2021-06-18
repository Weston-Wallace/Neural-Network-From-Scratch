import numpy as np


def linear(x):
    return x


def relu(x):
    return np.maximum(0, x)


def sigmoid(x):
    return np.divide(1, np.add(1, np.exp(np.negative(x).tolist())))


def softmax(x):
    return np.divide(np.exp(np.subtract(x, np.max(x)).tolist()), np.sum(np.exp(np.subtract(x, np.max(x)).tolist()), keepdims=True))


def cross_entropy_loss(predicted, target):
    return np.negative(np.sum(np.multiply(np.log(np.clip(predicted, 1e-14, 1 - 1e-14)), target)))


def categorical_cross_entropy_loss(predicted, target):
    return np.negative(np.log(np.clip(predicted, 1e-14, 1 - 1e-14)[np.argmax(target)]))


def sparse_categorical_cross_entropy_loss(predicted, target):
    return np.negative(np.log(np.clip(predicted, 1e-14, 1 - 1e-14)[target]))


func_dict = {
    "linear": linear,
    "relu": relu,
    "sigmoid": sigmoid,
    "softmax": softmax,
    "cross entropy loss": cross_entropy_loss,
    "cel": cross_entropy_loss,
    "categorical cross entropy loss": categorical_cross_entropy_loss,
    "ccel": categorical_cross_entropy_loss,
    "sparse categorical cross entropy loss": sparse_categorical_cross_entropy_loss,
    "sccel": sparse_categorical_cross_entropy_loss
}

dict_func = {
    linear: "linear",
    relu: "relu",
    sigmoid: "sigmoid",
    softmax: "softmax",
    cross_entropy_loss: "cel",
    categorical_cross_entropy_loss: "ccel",
    sparse_categorical_cross_entropy_loss: "sccel"
}
