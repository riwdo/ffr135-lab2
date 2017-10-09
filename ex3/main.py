import numpy as np
import matplotlib.pyplot as plt
import sys

N_NEURONS = 4
LEARNING_RATE = 0.02
N_UPDATES = 1000


def get_data(lines):
    temp = []
    Y = []
    for l in lines:
        temp_b = l.split("\t")
        first = float(temp_b[0])
        second = float(temp_b[1])
        third = float(temp_b[2])
        Y.append(first)
        temp.append(np.array([second, third]).T)
    return np.asarray(Y), np.asarray(temp)


def generate_weights(n_neurons, n_cols):
    return np.random.rand(n_neurons, n_cols) * 2 - 1


def random_pattern_index(size):
    return np.random.randint(0, size)


def calc_distance(point, neuron):
    return np.linalg.norm(point - neuron)


def activation(pattern_index, weights):
    denominator = 0
    for i in range(0, len(weights)):
        denominator += ((np.exp((-(calc_distance(patterns[pattern_index], weights[i]))**2))) / 2)

    max_g = 0
    winning_index = 0
    for index in range(0, N_NEURONS):
        g = ((np.exp((-(calc_distance(patterns[pattern_index], weights[index])) ** 2))) / 2) / denominator
        if g >= max_g:
            max_g = g
            winning_index = index
    return max_g, winning_index


f = open('data_ex2_task3_2017.txt', 'r')
data_lines = f.readlines()
f.close()
Y, patterns = get_data(data_lines)
weights = generate_weights(N_NEURONS, 2)
g = []

print weights
for i in range(0, N_UPDATES):
    rpi = random_pattern_index(len(patterns))
    g_i, winning_index = activation(random_pattern_index(len(patterns)), weights)
    weights[winning_index] += LEARNING_RATE * (patterns[rpi] - weights[winning_index])
    g.append(g_i)
    sys.stdout.write("ORDER: Running epoch %d of %d...\r" % (i + 1, N_UPDATES))
    sys.stdout.flush()
g = g[-N_NEURONS:]
print weights

# HERE COMES THE SUPERVISED LEARNING

exit()

arr_1 = []
arr_2 = []
for a in patterns:
    if a[0] == 1:
        arr_1.append((a[1], a[2]))
    else:
        arr_2.append((a[1], a[2]))
arr_1 = np.array(arr_1)
arr_2 = np.array(arr_2)

plt.scatter(arr_1[:, 0], arr_1[:, 1], c='r')
plt.scatter(arr_2[:, 0], arr_2[:, 1], c='b')
plt.show()
