import numpy as np
import matplotlib.pyplot as plt
import sys
from pprint import pprint

INPUT_SIZE = 1000
NEURON_SIZE = 100

TAO_SIGMA = 300
SIGMA_ZERO = 100

T_ORDER = 1
ETA_ZERO = .1


def gaussian_width(sigma_zero, t, tao_sigma):
    return sigma_zero * np.exp(-float(t) / tao_sigma)


def _generate_points(x1_low, x1_high, x2_low, x2_high, size):
    x1 = np.random.uniform(x1_low, x1_high, size)
    x2 = np.random.uniform(x2_low, x2_high, size)
    points = []
    for i in range(size):
        points.append([x1[i], x2[i]])
    return np.array(points)


def generate_neurons(size):
    return _generate_points(-1, 1, -1, 1, size)


def generate_points(size):
    iter_size = size // 3
    a = _generate_points(0, 0.5, 0, 0.5, iter_size)
    a = np.concatenate((a, _generate_points(0.5, 1, 0.5, 1, iter_size)))
    a = np.concatenate((a, _generate_points(0, 0.5, 0.5, 1, (size - (iter_size * 2)))))
    return a


def generate_weights(n_points, n_neurons):
    return np.random.rand(n_points, n_neurons) * 2 - 1


def calc_distance(point, neuron):
    return np.linalg.norm(point - neuron)


def neighbor_func(winning_dist, other_dist, sigma):
    return np.exp(np.square(-np.abs(other_dist - winning_dist)) / (2 * np.square(sigma)))


def calc_delta_weight(sigma, learning_rate, distance):
    return sigma * learning_rate * distance


def find_nearest_neuron(point_index, points, neurons_weights):
    min_distance = 1000000.0
    nearest_neuron_index = 0
    distances = []
    for i in range(0, len(neurons_weights)):
        dist = calc_distance(points[point_index], neurons_weights[i])
        if dist < min_distance:
            nearest_neuron_index = i
            min_distance = dist
        distances.append(dist)
        # print points[point_index], neurons_weights[i], calc_distance(points[point_index], neurons_weights[i])
    # print 'Nearest neuron:', nearest_neuron_index, '- Distance:', min_distance
    return nearest_neuron_index, np.array(distances)


points = generate_points(INPUT_SIZE)
np.random.shuffle(points)
neurons_W = generate_neurons(NEURON_SIZE)

for i in range(T_ORDER):
    sys.stdout.write("Epoch %d of %d\r" % (i, T_ORDER))
    sys.stdout.flush()
    learning_rate = gaussian_width(ETA_ZERO, i, TAO_SIGMA)
    sigma_rate = gaussian_width(SIGMA_ZERO, i, TAO_SIGMA)
    for j in range(INPUT_SIZE):
        winning_index, distances = find_nearest_neuron(j, points, neurons_W)
        for k in range(NEURON_SIZE):
            print neurons_W[k]
            neurons_W[k] += calc_delta_weight(learning_rate, neighbor_func(distances[winning_index], distances[k], sigma_rate), distances[k])
            print neurons_W[k]
        exit()
    # print learning_rate, sigma_rate

# exit()
plt.ion()
plt.scatter(points[:, 0], points[:, 1])
plt.scatter(neurons_W[:, 0], neurons_W[:, 1])
pprint(neurons_W)
# plt.scatter(points[0][0], points[0][1], c='r')
# plt.scatter(neurons_W[test_i][0], neurons_W[test_i][1], c='g')
plt.pause(2)
plt.ioff()
plt.show()
