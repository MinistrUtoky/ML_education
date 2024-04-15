import numpy as np
from matplotlib import pyplot as plt

class LinearRegression:
    def __init__(self):
        self.N = 1000
        self.x = np.linspace(0, 1, self.N)
        self.z = 20 * np.sin(2 * np.pi * 3 * self.x) + 100 * np.exp(self.x)
        self.error = 10 * np.random.randn(self.N)
        self.t = self.z + self.error

    def run(self):
        self.make_plot(1)
        self.make_plot(8)
        self.make_plot(100)
        plt.plot(np.linspace(1, 100, 100).T, self.E(1, 100))
        plt.show()

    def make_plot(self, M):
        plt.scatter(self.x, self.t)
        plt.plot(self.x, self.z, 'r')
        plt.plot(self.x, self.y(M), 'y')
        plt.show()

    def y(self, M):
        F = self.F(M)
        w = self.w(F)
        return np.dot(F, w)

    def w(self, F):
        print(np.dot(F.T, F).shape)
        return np.dot(np.dot(np.linalg.inv(np.dot(F.T, F)), F.T), self.t)

    def E(self, M0, M1):
        E = []
        for m in range(M0, M1+1):
            y = self.y(m)
            E.append(sum((self.t - y)**2))
        return 0.5*np.array(E)

    def F(self, M):
        F = []
        for xi in self.x:
            F.append(self.f(xi, M))
        return np.array(F)

    def f(self, x, M):
        f = []
        for m in range(M+1):
            f.append(x**m)
        return np.array(f)