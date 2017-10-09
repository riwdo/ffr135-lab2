import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

LEARNING_RATE = 0.001
updates = np.arange(0, 20000)


def generate_weights(n_points, n_neurons):
    return np.random.rand(n_points, n_neurons) * 2 - 1


def get_input_patterns(lines):
    a = []
    for l in lines:
        first = float(l.split("\t")[0])
        second = float(l.split("\t")[1])
        a.append(np.array([first, second]).T)
    return np.asarray(a)


def random_pattern_index(size):
    return np.random.randint(0, size)


def network_output(input_pattern, weights):
    return np.sum(np.multiply(weights, patterns[input_pattern]))


def update_weight(input_patterns, index, weights):
    output = network_output(index, weights)
    return LEARNING_RATE * output * (input_patterns[index] - output * weights)


def oja(input_data, weights):
    weight_modulus = []
    for i in range(len(updates)):
        index = random_pattern_index(len(input_data))
        weights += update_weight(input_data, index, weights)
        # print np.mean(np.dot(input_data, input_data.T))
        # print input_data.shape, input_data.T.shape
        # C = np.dot(input_data, input_data.T)
        # print C
        # exit()
        weight_modulus.append(np.linalg.norm(weights))
    return weights, weight_modulus


f = open('data_ex2_task2_2017.txt', 'r')
data_lines = f.readlines()
f.close()

patterns = get_input_patterns(data_lines)
norm_patterns = stats.zscore(patterns)

W = generate_weights(1, 2)
norm_W = generate_weights(1, 2)

W, W_mod = oja(patterns, W)
norm_W, norm_W_mod = oja(norm_patterns, norm_W)

f, axarr = plt.subplots(2, 2)
axarr[0, 0].set_title('Original Data')
axarr[0, 0].scatter(patterns[:, 0], patterns[:, 1], color='purple', s=2)
axarr[0, 0].plot([0, W[0, 0]], [0, W[0, 1]], color='red', linewidth=4)

axarr[1, 0].set_ylabel('|weights|')
axarr[1, 0].set_xlabel('iterations')
axarr[1, 0].scatter(updates, np.asarray(W_mod), s=2)

axarr[0, 1].set_title('Normalized Data')
axarr[0, 1].scatter(norm_patterns[:, 0], norm_patterns[:, 1], color='purple', s=2)
axarr[0, 1].plot([0, norm_W[0, 0]], [0, norm_W[0, 1]], color='red', linewidth=4)

axarr[1, 1].set_ylabel('|weights|')
axarr[1, 1].set_xlabel('iterations')
axarr[1, 1].scatter(updates, np.asarray(norm_W_mod), s=2)

plt.show()
