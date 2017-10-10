import numpy as np
import sys
import json

# Hyper parameters
learning_rate = .1
beta = .5
iterations = 3000

storage_training = []
storage_validation = []
storage_classification = []


# g(x) and g'(x)
def activation(x, deriv=False):
    if not deriv:
        return np.tanh(beta * x)
    else:
        return beta * (1 - np.tanh(beta * x) ** 2)


# sgn method for each element in numpy array
def sgn(z):
    sign = lambda x: -1 if x < 0 else 1
    shp = z.shape
    z = np.fromiter((sign(xi) for xi in z), z.dtype)
    z = np.reshape(z, shp)
    return z


def classification_error(x, y, weights):
    p = x.shape[0]
    y_val = activation(np.dot(x, weights))
    err = np.absolute(y - sgn(y_val))
    t = float(1) / (2 * p)
    return t * np.sum(err)


def energy_fn(x, y, w):
    y_val = activation(np.dot(x, w))
    err = y - y_val
    energy = np.sum(np.square(err) / 2)
    return energy


def train(X, Y, N, M):
    s_train = []
    s_class = []
    # Initialize weights with mean 0 in interval [-1, 1]
    weight_interval = [-1, 1]
    w = (weight_interval[1] - weight_interval[0]) * np.random.random((N, M)) + weight_interval[0]

    # Initialize bias with mean 0 in interval [-1, 1]
    bias_interval = [-1, 1]
    b = (bias_interval[1] - bias_interval[0]) * np.random.random((1, M)) + bias_interval[0]
    Y = Y.reshape(len(Y), 1)
    for iteration in xrange(iterations):

        # Get a random pattern and its associated target
        length = X.shape[0]
        rand_index = int(np.floor(np.random.random_sample() * length))
        x = X[rand_index, :].reshape(1, N)
        y_actual = Y[rand_index, :].reshape(M, 1)

        # feed forward
        ys = np.dot(x, w) + b
        y = activation(ys)

        delta_i = (y_actual - y) * activation(ys, deriv=True)
        delta_w = learning_rate * np.dot(x.T, delta_i)
        delta_b = learning_rate * delta_i

        b += delta_b
        w += delta_w
        if iteration % 100 == 0:
            s_train.append(energy_fn(X, Y, w))
            # s_val.append(energy_fn(X_val, Y_val, w, b))

            # sys.stdout.write(
            #     "[%d] Training set error: %f \t Validation set error: %f\r"
            #     % (iteration, classification_error(X, Y, w),
            #        classification_error(X_val, Y_val, w)))
            # sys.stdout.flush()
    print s_train
    print len(s_train)
    print ""

    # for experiment in range(10):
    #
    #     storage_classification.append([classification_error(X, Y, w)])
#
# with open(OUTFILE_TRAINING, 'w+') as f:
#     f.write(json.dumps(storage_training))
#
# with open(OUTFILE_VALIDATION, 'w+') as f:
#     f.write(json.dumps(storage_validation))
#
# with open(OUTFILE_CLASSFICATION_ERROR, 'w+') as f:
#     f.write(json.dumps(storage_classification))


