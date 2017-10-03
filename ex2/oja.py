import numpy as np
import matplotlib.pyplot as plt
import sys
from random import randint

learning_rate = 0.001
updates = 10**4


def generate_weights(n_points, n_neurons):
    return np.random.rand(n_points, n_neurons) * 2 - 1


def get_input_patterns(lines):
    patterns = []
    for l in lines:
        first = float(l.split("\t")[0])
        second = float(l.split("\t")[1])
        patterns.append(np.array([first, second]).T)
    return np.asarray(patterns)


def random_pattern_index(size):
    return np.random.randint(0,size)


def network_output(input_pattern):
    output = 0
    for i in range(len(patterns[input_pattern])):
        output += weights[i]*patterns[input_pattern]
    return output


def update_weight(size):
    index = random_pattern_index(size)
    output = network_output(index)
    print output
    print weights[index]
    weights[index] += learning_rate * output * (patterns[index]-output*weights[index])
    print weights[index]


f = open('data_ex2_task2_2017.txt','r')
data_lines = f.readlines()
f.close()

patterns = get_input_patterns(data_lines)
weights = generate_weights(401,2)
print weights[1]
plt.scatter(patterns[:,0], patterns[:,1])
plt.plot(weights[:,0], weights[:,1])
for i in range(updates):
    print i
    update_weight(len(patterns))
plt.plot(weights[:,0], weights[:,1])
plt.show()




