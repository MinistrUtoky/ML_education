import math
import random

import numpy as np
from matplotlib import pyplot as plt

class RegularizedRegression:
    def __init__(self):
        self.avaliable_funcs = [self.sin, self.cos, self.exp, self.sqrt, self.polynom]

        self.reg_coefs_lambda = [1e-6, 1e-5, 0.0001, 0.001, 0.01, 0.1, 0.5, 1, 5, 10, 50, 100, 500, 1000] # penalties?


    def monte_carlo(self, X, train_slice=0.60, test_slice=0.2):
        train_size = int(train_slice * len(X))
        test_size = int(test_slice * len(X))
        permutation = np.random.permutation(X)
        test = permutation[:test_size]
        train = permutation[test_size: train_size+test_size]
        validation = permutation[train_size+test_size:]
        return train, validation, test

    def monte_carlo_2(self, X, Y, train_slice=0.6, test_slice=0.2):
        return self.monte_carlo(
                   np.concatenate(
                   [X.copy().reshape(len(X), -1), Y.copy().reshape(len(Y), -1)],
                   axis=1),
                   train_slice, test_slice)

    def run(self, pumping_iterations, M):
        max_pow = M // (len(self.avaliable_funcs))
        self.all_func_variants = self.generate_all_func_variants(max_pow)
        N = 1000
        self.N = N
        self.M = M
        x = np.linspace(0, 1, N)
        z = 20 * np.sin(2 * np.pi * 3 * x) + 100 * np.exp(x)
        error = 10 * np.random.randn(N)
        t = z + error

        train_percentage = 60 #random.Random.randint(0, 100)
        validation_percentage = 20 #random.Random.randint(0, 100 - train_percentage)
        test_percentage = 100-train_percentage-validation_percentage

        XYZ = self.monte_carlo(
            np.concatenate(
                [x.copy().reshape(len(x), -1),
                 t.copy().reshape(len(t), -1),
                 z.copy().reshape(len(z), -1)],
                axis=1),
            train_percentage / 100.0, test_percentage / 100.0)

        XYZ_train = XYZ[0].T
        XYZ_validation = XYZ[1].T
        XYZ_test = XYZ[2]
        XYZ_test.sort(axis=0)
        XYZ_test = XYZ_test.T
        E_min = 1e+256
        best_w = np.zeros(M)
        best_b = np.zeros(M)
        best_l = self.reg_coefs_lambda[0]

        for i in range(pumping_iterations):
            b = np.random.permutation(self.all_func_variants)[:M]
            F = self.F(XYZ_train[0], b)
            for l in self.reg_coefs_lambda:
                w = self.w(l, F, XYZ_train[1])
                E = self.E(w, XYZ_validation[0], XYZ_validation[1], l, b)
                if E < E_min:
                    best_w = w
                    best_b = b
                    best_l = l
                    E_min = E

        E = self.E(best_w, XYZ_test[0], XYZ_test[1], best_l, best_b)
        print("Test error: " + str(E))
        print("Best coefficients: " + str(best_l))
        print("Best weights: ")
        print(best_w)
        print("Best basis: ")
        for bb in best_b:
            print(bb[0].__name__ + "**"+str(bb[1]), end=" ; ")
        predictions = self.test_with_the_best(x, best_b, best_w)
        plt.scatter(x, t)
        plt.plot(x, z, 'r')
        plt.plot(x, predictions, 'y')
        plt.show()

    def test_with_the_best(self, X, best_b, best_w):
        return self.y(self.F(X, best_b), best_w)

    def y(self, F, w):
        return F @ w

    def w(self, lamda, F, t):
        return np.linalg.pinv(F.T @ F + lamda*np.identity(F.shape[1])) @ F.T @ t

    def F(self, X, funcs):
        F = []
        for xi in X:
            F.append(self.f(xi, funcs))
        return np.array(F)

    def f(self, x, functions):
        f = []
        for m in range(self.M):
            f.append(functions[m][0](x, functions[m][1]))
        return f

    def E(self, w, x, t, l, funcs):
        return 1 / len(x) * np.array(sum((t - self.y(self.F(x, funcs), w)) ** 2)+l * sum(w**2))

    def sin(self, x, pow):
        return np.power(np.sin(x), pow)
    def cos(self, x, pow):
        return np.power(np.cos(x), pow)
    def exp(self, x, pow):
        return np.power(np.exp(x), pow)
    def sqrt(self, x, pow):
        return np.power(np.sqrt(x), pow)
    def polynom(self, x, pow):
        return np.power(x, pow)
    def random_func(self):
        return self.avaliable_funcs[random.randint(0, len(self.avaliable_funcs)-1)]
    def some_random_funcs(self):
        funcs = []
        for i in range(self.M):
            funcs.append(self.all_func_variants[i])
        return funcs

    def generate_all_func_variants(self, max_pow):
        funcs = []
        for i in range(len(self.avaliable_funcs)):
            for j in range(1, max_pow+1):
                funcs.append((self.avaliable_funcs[i], j))
        return funcs