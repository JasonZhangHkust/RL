# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt

mu, sigma = 0, 0.01

s = np.random.normal(mu, sigma, 10)

q = np.ones(10, dtype=float)

elipse = 0.1

plt.figure(1)

Q1 = np.zeros(10, dtype=float)
N = np.zeros(10, dtype=float)
R_Acc = 0
for i in range(1000):
    s = np.random.normal(mu, sigma, 10)
    q = q + s  # update the q*(a)
    if np.random.uniform() > elipse:
        A_index = np.argmax(Q1)
    else:
        A_index = np.random.randint(10)

    r_mean = q[A_index]
    R = np.random.normal(r_mean, 1)
    R_Acc = R_Acc + R
    N[A_index] = N[A_index] + 1
    Q1[A_index] = Q1[A_index] + (R - Q1[A_index])/N[A_index]
    plt.plot(i+1, R_Acc/(i+1), 'r-x')

plt.show()
