import numpy as np
import matplotlib.pyplot as plt

INPUT_SIZE = 1000
NEURON_SIZE = 100


def _generate_points(x1_low, x1_high, x2_low, x2_high, size):
    x1 = np.random.uniform(x1_low, x1_high, size)
    x2 = np.random.uniform(x2_low, x2_high, size)
    points = []
    for i in range(size):
        points.append([x1[i], x2[i]])
    return np.array(points)


def generate_neurons(size):
    return _generate_points(0, 1, 0, 1, size)


def generate_points(size):
    iter_size = size // 3
    a = _generate_points(0, 0.5, 0, 0.5, iter_size)
    a = np.concatenate((a, _generate_points(0.5, 1, 0.5, 1, iter_size)))
    a = np.concatenate((a, _generate_points(0, 0.5, 0.5, 1, (size - (iter_size * 2)))))
    return a


def generate_weights(n_points, n_neurons):
    return np.random.rand(n_points, n_neurons) * 2 - 1


def calc_distance(x1, y1, x2, y2):
    pass


def find_nearest_neuron(point_index, points, neurons, weights):
    for i in range(0, len(neurons)):
        print np.linalg.norm(points[point_index]-neurons[i])
        print points[point_index], neurons[i], weights[point_index][i]
        exit()


points = generate_points(INPUT_SIZE)
np.random.shuffle(points)
neurons = generate_neurons(NEURON_SIZE)
weights = generate_weights(INPUT_SIZE, NEURON_SIZE)
find_nearest_neuron(0, points, neurons, weights)

# exit()
plt.ion()
plt.scatter(points[:, 0], points[:, 1])
plt.scatter(neurons[:, 0], neurons[:, 1])
plt.pause(2)
plt.ioff()
plt.show()
