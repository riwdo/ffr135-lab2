import numpy as np
import matplotlib.pyplot as plt
import scipy
import sys
from random import randint

learning_rate = 0.001
#updates = 10**4

updates = [0]
for x in range(1, 20000):  # [1, 20, 40, 60, ..., 400]
    updates.append(x)

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
    return np.sum(np.multiply(weights,patterns[input_pattern]))


def update_weight(index):
    output = network_output(index)
    return learning_rate * output * (patterns[index]-output*weights)


f = open('data_ex2_task2_2017.txt','r')
data_lines = f.readlines()
f.close()

patterns = get_input_patterns(data_lines)
weights = generate_weights(1,2)
f, axarr = plt.subplots(2,sharex=False)
axarr[0].scatter(patterns[:,0], patterns[:,1],color='purple',s=2)
#axarr[0].scatter(weights[:,0], weights[:,1], color='blue',s=2)
#plt.plot(weights[:,0], weights[:,1])
weight_modulus = []
for i in range(len(updates)):
    print i
    index = random_pattern_index(len(patterns))
    weights += update_weight(index)
    weight_modulus.append(np.linalg.norm(weights))

#print np.asarray(weight_modulus).size
axarr[1].scatter(updates,np.asarray(weight_modulus),s=2)
axarr[0].plot([0,weights[0,0]], [0,weights[0,1]], color='red')
plt.show()




