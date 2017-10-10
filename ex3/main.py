import numpy as np
import matplotlib.pyplot as plt
import sys
import supervised_model as sm

N_NEURONS = 4
LEARNING_RATE = 0.02
N_UPDATES = 100000

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


def activation(pattern_index, w):
    denominator = 0
    x = patterns[pattern_index]
    for i in range(0, N_NEURONS):
        denominator += np.exp((-np.linalg.norm(x - w[i]) ** 2) / 2)
    g_ = []
    for i in range(0, N_NEURONS):
        g = np.exp((-np.linalg.norm(x - w[i]) ** 2) / 2) / denominator
        g_.append(g)
    return np.array(g_, dtype=np.float_)


f = open('data_ex2_task3_2017.txt', 'r')
data_lines = f.readlines()
f.close()
Y, patterns = get_data(data_lines)
weights = generate_weights(N_NEURONS, 2)

print weights
for index in range(0, N_UPDATES):
    rpi = random_pattern_index(len(patterns))
    winning_index = np.argmax(activation(rpi, weights))
    weights[winning_index] += LEARNING_RATE * (patterns[rpi] - weights[winning_index])
    sys.stdout.write("Running epoch %d of %d...\r" % (index + 1, N_UPDATES))
    sys.stdout.flush()
print ''
print weights

g_s = []
for i in range(0, len(patterns)):
    g_s.append(activation(i, weights))

# HERE COMES THE SUPERVISED LEARNING

sm.train(np.array(g_s), Y, N_NEURONS, 1)
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
