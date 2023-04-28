import numpy as np
import random
from typing import List, Tuple, Dict, Callable

random.seed(1234)


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
    return np.exp(Z) / sum(np.exp(Z))


def sigmoid(z: List) -> List:
    return 1.0 / (1 + np.exp(-z))


def forward_prop(x_data: List[List], w1: List, w2: List) -> Tuple:
    z1 = w1.dot(x_data.T)
    a1 = sigmoid(z1)
    z2 = w2.dot(a1)
    a2 = softmax(z2)
    return a1, a2


def calculate_error(y_data: List, a2: List) -> float:
    temp = np.subtract(1, a2)
    temp2 = np.log(temp, out=temp, where=temp > 0)
    error = (-1 / len(y_data)) * np.sum(y_data * np.log(a2.T) + np.subtract(1, y_data) * temp2.T)
    return error


def one_hot(a2: List) -> List:
    return np.identity(4)[np.argmax(a2, axis=0)]


def sigmoid_deriv(z: List) -> List:
    return z * (1 - z)


def backward_prop(a1: List, a2: List, w2: List, y_data: List) -> Tuple:
    dw2 = sigmoid_deriv(a2) * (y_data.T - a2)
    dw1 = sigmoid_deriv(a1) * w2.T.dot(dw2)
    return dw1, dw2


def update_params(w1: List, w2: List, dw1: List, dw2: List, a1: List, alpha: float, x_data: List[List]) -> Tuple:
    w1 += alpha * dw1.dot(x_data)
    w2 += alpha * dw2.dot(a1.T)
    return w1, w2


def learn_model(data: List[List], hidden_nodes: int, verbose=False) -> Tuple[List, List]:
    trans_data = transform_data(data)
    x_data = np.append(np.ones([len(data), 1]), trans_data[:, :-4], axis=1)
    y_data = trans_data[:, -4:]
    w1 = np.random.rand(hidden_nodes, x_data.shape[1])
    w2 = np.random.rand(len(y_data[0]), hidden_nodes)
    epsilon, alpha, previous_error = 1E-07, 0.01, 0.0
    current_error, count = 10, 0
    while abs(current_error - previous_error) > epsilon:
        a1, a2 = forward_prop(x_data, w1, w2)
        dw1, dw2 = backward_prop(a1, a2, w2, y_data)
        w1, w2 = update_params(w1, w2, dw1, dw2, a1, alpha, x_data)
        previous_error = current_error
        current_error = calculate_error(y_data, a2)
        if verbose and count % 1000 == 0:
            print("The current error is: " + str(current_error))
        if current_error > previous_error:
            alpha = alpha / 10
        count += 1
    return w1, w2


def apply_model(model: Tuple[List, List], test_data: List[List], labeled=False) -> List[List[Tuple]]:
    trans_data = transform_data(test_data)
    x_data = np.append(np.ones([len(test_data), 1]), trans_data[:, :-4], axis=1)
    y_data = trans_data[:, -4:]
    _, y_hat = forward_prop(x_data, model[0], model[1])
    y_ehat = one_hot(y_hat)
    if labeled:
        return format_output(y_data, y_ehat)
    else:
        return format_output(y_ehat, np.round(y_hat.T, 2))


def format_output(y_data1: List, y_data2: List) -> List[List[Tuple]]:
    results = []
    for i in range(len(y_data1)):
        temp = []
        for j in range(len(y_data1[i])):
            temp.append((y_data1[i, j], y_data2[i, j]))
        results.append(temp)
    return results


def evaluate(results: List[List[Tuple]]):
    count = 0
    n = len(results)
    for result in results:
        if (1.0, 1.0) in result:
            count += 1
    e_rate = round((1 - (count / n)), 4) * 100
    print("The error rate is: " + str(e_rate) + "%")


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

    train_data = generate_data(clean_data, 100)
    test_data = generate_data(clean_data, 100)
    model = learn_model(train_data, 2, True)
    results = apply_model(model, test_data, True)
    evaluate(results)
