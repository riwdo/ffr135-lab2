import numpy as np
import matplotlib.pyplot as plt
import sys
from random import randint

INPUT_SIZE = 1000
NEURON_SIZE = 100

TAO_SIGMA = 300
SIGMA_ORDER_ZERO = 100
T_ORDER = 1000
ETA_ORDER_ZERO = .1

T_CONV = 20000
SIGMA_CONV = .9
ETA_CONV = .01

PLOT_POINTS = True
PLOT_ION = False


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


def neighbor_func(winning_index, other_index, sigma):
    return np.exp((-((other_index - winning_index) ** 2) / (2 * (sigma ** 2))))


def calc_delta_weight(learning_rate, neighbor, distance):
    return learning_rate * neighbor * distance


def find_nearest_neuron(point_index, points, neurons_weights):
    min_distance = 1000000.0
    nearest_neuron_index = 0
    for i in range(0, len(neurons_weights)):
        dist = calc_distance(points[point_index], neurons_weights[i])
        if dist < min_distance:
            nearest_neuron_index = i
            min_distance = dist
    return nearest_neuron_index


def get_random_point_index(high):
    return randint(0, high - 1)


points = generate_points(INPUT_SIZE)
np.random.shuffle(points)
neurons_W = generate_neurons(NEURON_SIZE)

plt.figure(1)
plt.subplot(211)
plt.title('Ordering Phase')

if PLOT_ION:
    plt.ion()
    plt.scatter(points[:, 0], points[:, 1], c='b')
    plt.plot(neurons_W[:, 0], neurons_W[:, 1], 'go-', linewidth=1)
    plt.show()

for i in range(T_ORDER):
    sys.stdout.write("ORDER: Running epoch %d of %d...\r" % (i, T_ORDER))
    sys.stdout.flush()

    learning_rate = gaussian_width(ETA_ORDER_ZERO, i, TAO_SIGMA)
    sigma_rate = gaussian_width(SIGMA_ORDER_ZERO, i, TAO_SIGMA)

    random_point_index = get_random_point_index(INPUT_SIZE)
    winning_neuron_index = find_nearest_neuron(random_point_index, points, neurons_W)
    for j in range(NEURON_SIZE):
        neurons_W[j] += calc_delta_weight(learning_rate, neighbor_func(winning_neuron_index, j, sigma_rate), points[random_point_index] - neurons_W[j])

    if PLOT_ION:
        if i % 50 == 0:
            plt.clf()  # Clears plot before rendering
            if PLOT_POINTS:
                plt.scatter(points[:, 0], points[:, 1], c='b', s=8)
            plt.plot(neurons_W[:, 0], neurons_W[:, 1], 'go-', linewidth=1)
            plt.pause(0.1)

if PLOT_POINTS:
    plt.scatter(points[:, 0], points[:, 1], c='b', s=4)
plt.plot(neurons_W[:, 0], neurons_W[:, 1], 'go-', linewidth=1)

print ''

plt.subplot(212)
plt.title('Convergence Phase')

for i in range(T_CONV):
    sys.stdout.write("CONV: Running epoch %d of %d...\r" % (i, T_CONV))
    sys.stdout.flush()
    random_point_index = get_random_point_index(INPUT_SIZE)
    winning_neuron_index = find_nearest_neuron(random_point_index, points, neurons_W)
    for j in range(NEURON_SIZE):
        neurons_W[j] += calc_delta_weight(ETA_CONV, neighbor_func(winning_neuron_index, j, SIGMA_CONV), points[random_point_index] - neurons_W[j])

    if PLOT_ION:
        if i % 1000 == 0:
            plt.clf()  # Clears plot before rendering
            if PLOT_POINTS:
                plt.scatter(points[:, 0], points[:, 1], c='b')
            plt.plot(neurons_W[:, 0], neurons_W[:, 1], 'go-', linewidth=1)
            plt.pause(0.1)

if PLOT_ION:
    plt.ioff()
    plt.clf()
print ''

if PLOT_POINTS:
    plt.scatter(points[:, 0], points[:, 1], c='b', s=4)
plt.plot(neurons_W[:, 0], neurons_W[:, 1], 'go-', linewidth=1)
plt.show()
