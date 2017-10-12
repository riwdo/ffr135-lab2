import numpy as np
import matplotlib.pyplot as plt

x_vals = np.arange(0, 10) + 1
y_vals = np.array([0.5, 0.1205, 0.0683333333333, 0.032375, 0.0128, 0.00425, 0.00178571428571, 0.0009375, 0.000444444444444, 0.0003])

plt.xlabel('K')
plt.ylabel('Avg error')
plt.plot(x_vals, y_vals, c='black')
plt.show()
