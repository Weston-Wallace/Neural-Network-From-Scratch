import numpy as np


def load_data(file_path):
    with open(file_path, "rb") as tl_file:
        tl_file.read(3)
        ndim = int.from_bytes(tl_file.read(1), "big")
        lengths = [int.from_bytes(tl_file.read(4), "big") for _ in range(ndim)]
        data = np.array([i for i in tl_file.read()])
    return data.reshape(lengths)
