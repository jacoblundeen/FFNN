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


def softmax(Z: List) -> List:
    A = np.exp(Z) / sum(np.exp(Z))
    return A


def sigmoid(z):
    # temp = np.exp(-z, out=z, where=z < 1)
    return 1.0 / (1 + np.exp(-z))


def forward_prop(x_data: List[List], w1: List, w2: List) -> Tuple:
    z1 = w1.dot(x_data.T)
    a1 = sigmoid(z1)
    z2 = w2.dot(a1)
    a2 = softmax(z2)
    return z1, a1, z2, a2


def calculate_error(y_data: List, a2: List) -> float:
    n = len(y_data)
    temp = np.subtract(1, a2)
    temp2 = np.log(temp, out=temp, where=temp > 0)
    error = (-1 / n) * np.sum(y_data * np.log(a2.T) + np.subtract(1, y_data) * temp2.T)
    return error


def sigmoid_deriv(z):
    return z * (1 - z)


def backward_prop(z1, a1, z2, a2, w1, w2, x_data, y_data):
    m, _ = x_data.shape
    dw2 = sigmoid_deriv(a2) * (y_data.T - a2)
    dw1 = sigmoid_deriv(a1) * w2.T.dot(dw2)
    # db2 = 1 / m * np.sum(dz2, axis=1)
    # dz1 = w2.T.dot(dz2) * sigmoid_deriv(z1)
    # dw1 = 1 / m * dz1.dot(x_data)
    # db1 = 1 / m * np.sum(dz1, axis=1)
    return dw1, dw2


def update_params(w1, w2, dw1, dw2, a1, a2, alpha, x_data):
    w1 += alpha * dw1.dot(x_data)
    # b1 -= alpha * np.expand_dims(db1, axis=1)
    w2 += alpha * dw2.dot(a1.T)
    # b2 -= alpha * np.expand_dims(db2, axis=1)
    return w1, w2


def learn_model(data, hidden_nodes, verbose=False):
    x_data = np.append(np.ones([len(data), 1]), data[:, :-4], axis=1)
    y_data = data[:, -4:]
    m, n = x_data.shape
    num_classes = len(y_data[0])
    w1 = np.random.rand(hidden_nodes, n)
    # b1 = np.random.rand(hidden_nodes, 1)
    w2 = np.random.rand(num_classes, hidden_nodes)
    # b2 = np.random.rand(num_classes, 1)
    epsilon, alpha, previous_error = 1E-07, 0.01, 0.0
    current_error = 10
    count = 0
    while abs(current_error - previous_error) > epsilon:
        z1, a1, z2, a2 = forward_prop(x_data, w1, w2)
        dw1, dw2 = backward_prop(z1, a1, z2, a2, w1, w2, x_data, y_data)
        w1, w2 = update_params(w1, w2, dw1, dw2, a1, a2, alpha, x_data)
        previous_error = current_error
        current_error = calculate_error(y_data, a2)
        if verbose and count % 1000 == 0:
            print("The current error is: " + str(current_error))
        if current_error > previous_error:
            alpha = alpha / 10
        count += 1
    print(count)
    return a2


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
    model = learn_model(datum, 2, True)
    print(model)
