import numpy as np
from NeuralNetwork import Functions


class MLP:
    def __init__(self, shape, hidden_activation_func="linear", output_activation_func="linear"):
        self.shape = np.array(shape)
        self.weights = np.array(
            [np.random.randn(self.shape[i], self.shape[i + 1]) for i in np.arange(len(self.shape) - 1)],
            dtype=object) * 0.001
        self.biases = np.array(
            [np.random.randn(self.shape[i]) for i in np.arange(1, len(self.shape))],
            dtype=object) * 0.01
        self.out = np.empty(self.shape[-1])
        self.hidden_activation_func = Functions.func_dict[hidden_activation_func]
        self.output_activation_func = Functions.func_dict[output_activation_func]

    def set_hidden_activation_func(self, func):
        self.hidden_activation_func = Functions.func_dict[func]

    def set_output_activation_func(self, func):
        self.output_activation_func = Functions.func_dict[func]

    def set_params(self, weights, biases):
        self.weights = weights
        self.biases = biases

    def forward(self, inp):
        if len(self.shape) == 2:
            self.out = self.output_activation_func(np.dot(inp, self.weights[0]) + self.biases[0])
        else:
            current_hidden_value = self.hidden_activation_func(
                np.dot(inp, self.weights[0]) + self.biases[0])
            for i in np.arange(1, len(self.shape) - 2):
                current_hidden_value = np.dot(current_hidden_value, self.weights[i]) + self.biases[i]
            self.out = self.output_activation_func(
                np.dot(current_hidden_value, self.weights[-1]) + self.biases[-1])
        return self.out

    def make_dict(self):
        return {
            "type": "MLP",
            "shape": self.shape.tolist(),
            "weights": [i.tolist() for i in self.weights],
            "biases": self.biases.tolist(),
            "hidden function": Functions.dict_func[self.hidden_activation_func],
            "output function": Functions.dict_func[self.output_activation_func]
        }

    def copy(self):
        new_mlp = MLP(self.shape,
                      Functions.dict_func[self.hidden_activation_func],
                      Functions.dict_func[self.output_activation_func]
                      )
        new_mlp.set_params(self.weights.copy(), self.biases.copy())
        return new_mlp

    def update_weights(self, delta):
        self.weights += delta
        return self

    def update_biases(self, delta):
        self.biases += delta
        return self


# noinspection PyAttributeOutsideInit
class MLPBuilder:
    def __init__(self):
        self._hidden = "linear"
        self._out = "linear"

    def build(self):
        return MLP(self._shape, self._hidden, self._out)

    def shape(self, shape):
        self._shape = shape
        return self

    def hidden_activation_func(self, hidden_activation_func):
        self._hidden = hidden_activation_func
        return self

    def output_activation_func(self, output_activation_func):
        self._out = output_activation_func
        return self
