import numpy as np
import json
from NeuralNetwork import Functions
from NeuralNetwork import MLPBuilder


class AI:
    def __init__(self, nn, training_method, loss_method):
        self.nn = nn
        self.training_method = training_method
        self.loss_method = Functions.func_dict[loss_method]

    def train(self, X, y, learning_rate=0.005, epochs=100):
        print("Starting training...")
        X = np.array(X)
        n = 1
        for epoch in range(epochs):
            for batch, targets in zip(X, y):
                if self.training_method.casefold() == "evolutionary":
                    rng = np.random.default_rng()
                    candidates = [self.nn.copy().update_weights((learning_rate * 2) * np.array(
                        [rng.random((self.nn.shape[i], self.nn.shape[i + 1])) for i in
                         np.arange(len(self.nn.shape) - 1)]) - learning_rate).update_biases(
                        (learning_rate * 2) * np.array(
                            [rng.random(self.nn.shape[i]) for i in np.arange(1, len(self.nn.shape))]) + learning_rate)
                                  for _ in
                                  np.arange(99)]
                    candidates.append(self.nn)
                    preds = [[c.forward(i) for i in batch] for c in candidates]
                    loss = [np.mean([self.loss_method(p, target) for p, target in zip(pred, targets)]) for pred in preds]
                    best_loss = np.argmin(loss)
                    self.nn = candidates[best_loss]
                    print("Best loss for batch", n, "is", loss[best_loss])
                    n += 1
        print("Training finished; saving network")
        self.save("Evolution Guy 2")

    def save(self, file_name):
        d = self.nn.make_dict()
        d["training method"] = self.training_method
        d["loss method"] = Functions.dict_func[self.loss_method]
        with open("%s.json" % file_name, "w") as json_file:
            json.dump(d, json_file)

    @staticmethod
    def load(file_name):
        with open("%s.json" % file_name, "r") as json_file:
            json_obj = json.load(json_file)
        if json_obj["type"].casefold() == "mlp":
            nn = MLPBuilder() \
                .shape(json_obj["shape"]) \
                .hidden_activation_func(json_obj["hidden function"]) \
                .output_activation_func(json_obj["output function"]) \
                .build()
        else:
            nn = MLPBuilder() \
                .shape(json_obj["shape"]) \
                .hidden_activation_func(json_obj["hidden function"]) \
                .output_activation_func(json_obj["output function"]) \
                .build()
        return AIBuilder() \
            .nn(nn) \
            .training_method(json_obj["training method"]) \
            .loss_method(json_obj["loss method"]) \
            .build()

    @staticmethod
    def reshape_data(X, epochs, batch_size):
        assert len(X) % batch_size == 0
        new_shape = list(X.shape[1:])
        new_shape.insert(0, epochs)
        new_shape.insert(1, batch_size)
        return np.reshape(X, new_shape)


# noinspection PyAttributeOutsideInit
class AIBuilder:
    def build(self):
        return AI(self._nn, self._training_method, self._loss_method)

    def nn(self, nn):
        self._nn = nn
        return self

    def training_method(self, training_method):
        self._training_method = training_method
        return self

    def loss_method(self, loss_method):
        self._loss_method = loss_method
        return self
