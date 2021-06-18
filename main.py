import time
import multiprocessing

from NeuralNetwork import MNISTDataReader
from NeuralNetwork import MLPBuilder

if __name__ == '__main__':
    train_images = MNISTDataReader.load_data("NeuralNetwork/mnist data/train-images.idx3-ubyte")
    train_images = train_images.reshape(1000, 60, 784)
    # train_labels = MNISTDataReader.load_data("NeuralNetwork/mnist data/train-labels.idx1-ubyte")
    #
    # ai = AIBuilder() \
    #     .nn(MLPBuilder()
    #         .shape((784, 10, 10, 10))
    #         .output_activation_func("softmax")
    #         .hidden_activation_func("relu")
    #         .build()
    #         ) \
    #     .loss_method("sccel") \
    #     .training_method("evolutionary") \
    #     .build()
    #
    # train_labels = ai.reshape_data(train_labels, 1000, 60)
    #
    # ai.train(train_images, train_labels)

    nn = MLPBuilder()\
        .shape((784, 1000, 100, 10))\
        .output_activation_func("softmax")\
        .hidden_activation_func("relu")\
        .build()

    start = time.perf_counter()

    for batch in train_images:
        with multiprocessing.Pool() as pool:
            output = pool.map(nn.forward, batch)

    end = time.perf_counter()
    print(f"Time taken was {end - start} seconds.")
    start = time.perf_counter()

    output = [map(nn.forward, batch) for batch in train_images]

    end = time.perf_counter()
    print(f"Time taken was {end - start} seconds.")
