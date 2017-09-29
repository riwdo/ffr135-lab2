import numpy as np
import matplotlib.pyplot as plt

def _generate_points(x1_low, x1_high, x2_low, x2_high, size):
    x1 = np.random.uniform(x1_low,x1_high,size)
    x2 = np.random.uniform(x2_low,x2_high,size)
    points = []
    for i in range(size):
        points.append([x1[i],x2[i]])
    return np.array(points)

def generate_neurons(size):
    return _generate_points(0, 1, 0, 1, size)

def generate_points(size):
    iter_size = size // 3
    a = _generate_points(0, 0.5, 0, 0.5, iter_size)
    a = np.concatenate((a, _generate_points(0.5, 1, 0.5, 1, iter_size)))
    a = np.concatenate((a, _generate_points(0, 0.5, 0.5, 1, (size - (iter_size * 2)))))
    return a

points = generate_points(1000)
neurons = generate_neurons(100)

plt.scatter(points[:, 0], points[:, 1])
plt.scatter(neurons[:, 0], neurons[:, 1])
plt.show()
