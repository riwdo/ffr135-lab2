import numpy as np
import matplotlib.pyplot as plt


def get_data(lines):
    temp = []
    for l in lines:
        temp_b = l.split("\t")
        first = float(temp_b[0])
        second = float(temp_b[1])
        third = float(temp_b[2])
        temp.append(np.array([first, second, third]).T)
    return np.asarray(temp)


f = open('data_ex2_task3_2017.txt', 'r')
data_lines = f.readlines()
f.close()
data = get_data(data_lines)

arr_1 = []
arr_2 = []
for a in data:
    if a[0] == 1:
        arr_1.append((a[1], a[2]))
    else:
        arr_2.append((a[1], a[2]))
arr_1 = np.array(arr_1)
arr_2 = np.array(arr_2)

plt.scatter(arr_1[:, 0], arr_1[:, 1], c='r')
plt.scatter(arr_2[:, 0], arr_2[:, 1], c='b')
plt.show()
