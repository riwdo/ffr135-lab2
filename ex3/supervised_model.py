import numpy as np
import sys

# Hyper parameters
learning_rate = .1
beta = .5
iterations = 3000


def activation(x, deriv=False):
    if not deriv:
        return np.tanh(beta * x)
    else:
        return beta * (1 - np.tanh(beta * x) ** 2)


def sgn(z):
    sign = lambda x: -1 if x < 0 else 1
    shp = z.shape
    z = np.fromiter((sign(xi) for xi in z), z.dtype)
    z = np.reshape(z, shp)
    return z


class SupervisedModel:
    def __init__(self, n, m, w=None):
        weight_interval = [-1, 1]
        self.n = n
        self.m = m
        if w is None:
            self.w = (weight_interval[1] - weight_interval[0]) * np.random.random((n, m)) + weight_interval[0]
        else:
            self.w = w

    def classification_error(self, x, y):
        p = x.shape[0]
        y_val = activation(np.dot(x, self.w))
        err = np.absolute(y - sgn(y_val))
        t = float(1) / (2 * p)
        return t * np.sum(err)

    def train(self, X, Y):
        # Initialize bias with mean 0 in interval [-1, 1]
        bias_interval = [-1, 1]
        b = (bias_interval[1] - bias_interval[0]) * np.random.random((1, self.m)) + bias_interval[0]
        Y = Y.reshape(len(Y), 1)

        for iteration in xrange(iterations):

            # Get a random pattern and its associated target
            length = X.shape[0]
            rand_index = int(np.floor(np.random.random_sample() * length))
            x = X[rand_index, :].reshape(1, self.n)
            y_actual = Y[rand_index, :].reshape(self.m, 1)

            # feed forward
            ys = np.dot(x, self.w) + b
            y = activation(ys)

            delta_i = (y_actual - y) * activation(ys, deriv=True)
            delta_w = learning_rate * np.dot(x.T, delta_i)
            delta_b = learning_rate * delta_i

            b += delta_b
            self.w += delta_w

            sys.stdout.write("Running supervised training epoch %d of %d...\r" % (iteration + 1, iterations))
            sys.stdout.flush()
        print ''

    def valid(self, X, Y):
        error = 0
        for i in range(0, len(X)):
            error += self.classification_error(X[i], Y[i])
        return error / len(X)

    def get_guess(self, x):
        return activation(np.dot(x, self.w))[0]
