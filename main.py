import numpy as np
import random
from typing import List, Tuple, Dict, Callable

random.seed(49)


def blur(data):
    def apply_noise(value):
        if value < 0.5:
            v = random.gauss(0.10, 0.05)
            if v < 0.0:
                return 0.0
            if v > 0.75:
                return 0.75
            return v
        else:
            v = random.gauss(0.90, 0.10)
            if v < 0.25:
                return 0.25
            if v > 1.00:
                return 1.00
            return v

    noisy_readings = [apply_noise(v) for v in data[0:-1]]
    return noisy_readings + [data[-1]]


def generate_data(data, n):
    labels = list(data.keys())
    result = []

    # create n "label" and one hot encode
    for label in labels:
        for _ in range(n):
            datum = blur(random.choice(data[label]))
            xs = datum[0:-1]
            if label == "hills":
                result.append((xs, [1, 0, 0, 0]))
            elif label == "swamp":
                result.append((xs, [0, 1, 0, 0]))
            elif label == "forest":
                result.append((xs, [0, 0, 1, 0]))
            else:
                result.append((xs, [0, 0, 0, 1]))
    random.shuffle(result)
    return result


def learn_model(data, hidden_nodes, verbose=False):
    x_data = data[:, :-4]
    y_data = data[:, -4:]
    num_classes = len(y_data[0])
    w1 = np.random.rand(hidden_nodes)
    b1 = np.random.rand(hidden_nodes)
    w2 = np.random.rand(num_classes)
    b2 = np.random.rand(num_classes)
    epsilon, alpha, previous_error = 1E-07, 0.1, 0.0
    while abs(current_error - previous_error) > epsilon:
    pass


# This function is to convert the data structure from generate_data() into an easier to use numpy matrix
def transform_data(data: List[Tuple[List, List]]) -> List:
    datum = np.empty(len(data[0][0]) + 1)
    count = 0
    # Since the structure of data is a List[Tuple[List, List]], we have to loop over every tuple, combine the elements,
    # and build the matrix row by row.
    for result in data:
        obs = result[0] + result[1]
        if count == 0:
            datum = obs
        else:
            datum = np.vstack((datum, obs))
        count += 1
    return datum


if __name__ == "__main__":
    clean_data = {
        "plains": [
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, "plains"]
        ],
        "forest": [
            [0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, "forest"],
            [0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, "forest"],
            [1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, "forest"],
            [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, "forest"]
        ],
        "hills": [
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, "hills"],
            [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, "hills"],
            [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, "hills"],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, "hills"]
        ],
        "swamp": [
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, "swamp"],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, "swamp"]
        ]
    }

    data = generate_data(clean_data, 100)
    datum = transform_data(data)
    model = learn_model(datum, 2)
    print(datum)
