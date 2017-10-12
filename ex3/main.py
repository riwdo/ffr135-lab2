import numpy as np
import matplotlib.pyplot as plt
import sys
import supervised_model

N_NEURONS = 1
LEARNING_RATE = 0.02
N_UPDATES = 100000
EXPERIMENTS = 20
PIXEL_PER_RANGE = 50


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


def activation(x, w):
    denominator = 0
    for i in range(0, N_NEURONS):
        denominator += np.exp((-np.linalg.norm(x - w[i]) ** 2) / 2)
    g_ = []
    for i in range(0, N_NEURONS):
        g = np.exp((-np.linalg.norm(x - w[i]) ** 2) / 2) / denominator
        g_.append(g)
    return np.array(g_, dtype=np.float_)


def shuffle_data(X, Y):
    if len(X) != len(Y):
        raise IndexError('X and Y does need to have same length')
    a = np.arange(0, len(X))
    np.random.shuffle(a)

    Y_rand, X_rand = [], []
    for i in a:
        X_rand.append(X[i])
        Y_rand.append(Y[i])
    return np.array(X_rand), np.array(Y_rand)


f = open('data_ex2_task3_2017.txt', 'r')
data_lines = f.readlines()
f.close()

best_error = 1
best_weight_supervised = None
best_weight_unsupervised = None

for experiment in range(0, EXPERIMENTS):
    print 'Running experiment %d by %d\n' % (experiment + 1, EXPERIMENTS)
    Y, patterns = get_data(data_lines)

    # Unsupervised simple competitive learning
    weights = generate_weights(N_NEURONS, 2)

    for index in range(0, N_UPDATES):
        rpi = random_pattern_index(len(patterns))
        winning_index = np.argmax(activation(patterns[rpi], weights))
        weights[winning_index] += LEARNING_RATE * (patterns[rpi] - weights[winning_index])
        sys.stdout.write("Running unsupervised training epoch %d of %d...\r" % (index + 1, N_UPDATES))
        sys.stdout.flush()
    print ''

    g_s = []
    for i in range(0, len(patterns)):
        g_s.append(activation(patterns[i], weights))

    X_new, Y_new = shuffle_data(g_s, Y)

    # Supervised simple perceptron network
    sm = supervised_model.SupervisedModel(N_NEURONS, 1)
    sm.train(X_new, Y_new)
    experiment_error = sm.valid(X_new, Y_new)
    print '\n'

    if experiment_error < best_error:
        # Store best experiment and it's result
        best_error = experiment_error
        best_weight_supervised = sm.w
        best_weight_unsupervised = weights

print 'Best avg classification error:', best_error
print 'Best supervised weights:', best_weight_supervised
print 'Best unsupervised weights:', best_weight_unsupervised

# Plotting
Y, patterns = get_data(data_lines)
input_pos = []
input_neg = []
for i in range(0, len(patterns)):
    if Y[i] == 1:
        input_pos.append(patterns[i])
    else:
        input_neg.append(patterns[i])
input_pos = np.array(input_pos)
input_neg = np.array(input_neg)

x_max, x_min = np.amax(patterns[:, 0]), np.amin(patterns[:, 0])
y_max, y_min = np.amax(patterns[:, 1]), np.amin(patterns[:, 1])

test_X = np.linspace(x_min, x_max, PIXEL_PER_RANGE)
test_Y = np.linspace(y_min, y_max, PIXEL_PER_RANGE)
plot_matrix = np.empty([PIXEL_PER_RANGE, PIXEL_PER_RANGE])

best_sm = supervised_model.SupervisedModel(N_NEURONS, 1, best_weight_supervised)
color_map = plt.get_cmap('binary')

for i in range(0, PIXEL_PER_RANGE):
    for j in range(0, PIXEL_PER_RANGE):
        uns = activation(np.array([test_X[i], test_Y[j]]), best_weight_unsupervised)
        guess = best_sm.get_guess(uns)
        plt.scatter(test_X[i], test_Y[j], color=color_map((guess+1) / 2), s=200)

plt.scatter(input_pos[:, 0], input_pos[:, 1], c='g')
plt.scatter(input_neg[:, 0], input_neg[:, 1], c='y')

plt.axis([x_min, x_max, y_min, y_max])
plt.show()
