# from sklearn.datasets import load_boston
from sklearn.datasets import fetch_california_housing
import math

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt


class GradientRegression:
    learning_rates = [1e-6, 1e-5, 0.0001, 0.001, 0.01]
    avg = 0
    std = 0.1
    lambdas = [0.01, 0.1, 0.5, 1, 5, 10, 50, 100, 500, 1000, 5000]
    difference_threshold = 0.01
    gradient_threshold = 1
    validation_iterations = 20
    X: np.hstack
    Y: np.ndarray

    def __init__(self):
        self.avaliable_funcs = [self.polynom]#[self.sin, self.cos, self.exp, self.sqrt, self.polynom]
        self.fetch_data()

    def fetch_data(self):
        data_url = "http://lib.stat.cmu.edu/datasets/boston"
        raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)
        #california = fetch_california_housing()
        #self.X = california.data
        #self.Y = california.target
        self.X = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
        self.Y = raw_df.values[1::2, 2]

    def monte_carlo(self, X, train_slice=0.60, test_slice=0.2):
        train_size = int(train_slice * len(X))
        test_size = int(test_slice * len(X))
        permutation = np.random.permutation(X)
        test = permutation[:test_size].T
        train = permutation[test_size: train_size + test_size].T
        validation = permutation[train_size + test_size:].T
        Xtrain, Ytrain = train[:-1], train[-1]
        Xvalidation, Yvalidation = validation[:-1], validation[-1]
        Xtest, Ytest = test[:-1], test[-1]
        return np.array([Xtrain, Ytrain]), np.array([Xvalidation, Yvalidation]), np.array([Xtest, Ytest])

    def monte_carlo_2(self, X, Y, train_slice=0.6, test_slice=0.2):
        return self.monte_carlo(
            np.concatenate(
                (X.copy().reshape(len(X), -1), Y.copy().reshape(len(Y), -1)),
                axis=1),
            train_slice, test_slice)

    def XY_train_validation_test(self):
        train_percentage = 60 #random.Random.randint(0, 100)
        validation_percentage = 20 #random.Random.randint(0, 100 - train_percentage)
        test_percentage = 100-train_percentage-validation_percentage

        XY = self.monte_carlo_2(self.X, self.Y, train_percentage/100, test_percentage/100)
        XY_train = XY[0]
        XY_validation = XY[1]
        XY_test = XY[2]
        return XY_train, XY_validation, XY_test

    def run(self, iters=100):
        self.all_func_variants = self.generate_all_func_variants(math.ceil(self.X.shape[1] / (len(self.avaliable_funcs))))
        XY_train, XY_validation, XY_test = self.XY_train_validation_test()
        Xtrain, Ytrain = XY_train[0].T, XY_train[1].T
        Xval, Yval = XY_validation[0].T, XY_validation[1].T
        Xtest, Ytest = XY_test[0].T, XY_test[1].T

        X_train_standardized = (Xtrain - Xtrain.mean()) / Xtrain.std()
        X_test_standardized = (Xtest - Xtest.mean()) / Xtest.std()
        X_val_standardized = (Xval - Xval.mean()) / Xval.std()

        for learning_rate in self.learning_rates:
            w = np.random.normal(0, 0.1, (1, self.X.shape[1]))[0]
            weight, Es = self.regress(X_train_standardized, Ytrain, w, self.lambdas[0], learning_rate, iters)
            print(f"Train error:"
                  f"{self.E(weight, X_train_standardized, Ytrain, self.lambdas[0], self.y(X_train_standardized, weight))} "
                  f"on trained model")
            print(f"Test error:"
                  f"{self.E(weight, X_test_standardized, Ytest, self.lambdas[0], self.y(X_test_standardized, weight))} "
                  f"on trained model")
            plt.title(f"RR with GD (rate: {learning_rate})")
            #self.make_errors_plot(Es)

            # second part with polynomial bases and validation
            E_min = 1e+256
            best_w = np.zeros(self.X.shape[1])
            best_b = np.random.permutation(self.all_func_variants)[:self.X.shape[1]]
            best_l = self.lambdas[0]
            for k in range(self.validation_iterations):
                w = np.random.normal(0, 0.1, (1, self.X.shape[1]))[0]
                b = np.random.permutation(self.all_func_variants)[:self.X.shape[1]]
                Xfuncs_train = self.F(X_train_standardized, b)
                Xfuncs_train = (Xfuncs_train - Xfuncs_train.mean()) / Xfuncs_train.std()
                Xfuncs_val = self.F(X_val_standardized, b)
                Xfuncs_val = (Xfuncs_val - Xfuncs_val.mean()) / Xfuncs_val.std()
                for lamda in self.lambdas:
                    weight2, _ = self.regress(Xfuncs_train, Ytrain, w, lamda, learning_rate, iters)
                    valE = self.E(weight2, Xfuncs_val, Yval, lamda, self.y(Xfuncs_val, weight2))
                    if valE < E_min:
                        best_w = weight2
                        best_b = b
                        best_l = lamda
                        E_min = valE
            Xfuncs_train = self.F(X_train_standardized, best_b)
            Xfuncs_train = (Xfuncs_train - Xfuncs_train.mean()) / Xfuncs_train.std()
            _, Es2 = self.regress(X_train_standardized, Ytrain, best_w, best_l,
                                                      learning_rate, iters)
            print(
                f"Train error for validated version:"
                f"{self.E(best_w, X_train_standardized, Ytrain, best_l, self.y(Xfuncs_train, best_w))} "
                f"on trained model")
            Xfuncs_test = self.F(X_test_standardized, best_b)
            print(
                f"Test error for validated version:"
                f"{self.E(best_w, X_test_standardized, Ytest, best_l, self.y(X_test_standardized, best_w))} "
                f"on trained model")
            plt.title(f"RR with GD (lambda : {self.lambdas[0]}; best: {best_l}) (learning rate: {learning_rate})")
            self.make_comparative_plot(list(range(0, len(Es))), list(range(0, len(Es2))), Es, Es2)

    def regress(self, Xtrain, Ytrain, w, lamda, learning_rate, iters):
        Es = []
        for i in range(iters):
            Ynew = self.y(Xtrain, w, 0)
            Es.append(self.E(w, Xtrain, Ytrain, lamda, Ynew))
            #print(f"Train cost:{Es[i]} \t iteration: {i}")
            g = self.gradient(Ytrain, Xtrain, w, lamda)
            if np.linalg.norm(learning_rate*g) < self.difference_threshold \
                    or np.linalg.norm(g) < self.gradient_threshold:
                break
            w = w - learning_rate * g
        return w, Es

    def make_errors_plot(self, errors):
        plt.plot(list(range(0, len(errors))), errors, color='b', label='error')
        plt.xlabel("iteration")
        plt.ylabel("error")
        plt.legend()
        plt.show()

    def make_comparative_plot(self, X1, X2, Y1, Y2):
        plt.plot(X1, Y1, label="Without validation")
        plt.plot(X2, Y2, color='r', label='With validation')
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.legend()
        plt.show()

    def y(self, F, w, b=0):
        return F @ w

    def w(self, lamda, F, t):
        return np.linalg.pinv(F.T @ F + lamda*np.identity(F.shape[1])) @ F.T @ t

    def F(self, X, funcs):
        F = []
        for xi in X:
            F.append(self.f(xi, funcs))
        F = np.array(F)
        return F

    def f(self, x, functions):
        f = []
        for m in range(len(functions)):
            f.append(functions[m][0](x[m], functions[m][1]))
        return f

    def E(self, w, x, t, lamda, y):
        return 1 / len(x) * np.array(sum((t - y) ** 2) + lamda * sum(w**2))

    def gradient(self, t, F, w, lamda):
        return -(t.T @ F).T + (w.T @ (F.T @ F)).T + lamda * w.T

    def sin(self, x, pow):
        return np.power(np.sin(x), pow)
    def cos(self, x, pow):
        return np.power(np.cos(x), pow)
    def exp(self, x, pow):
        return np.power(np.exp(x), pow)
    def sqrt(self, x, pow):
        return np.power(np.sqrt(abs(x)), pow)
    def polynom(self, x, pow):
        return np.power(x, pow)

    def generate_all_func_variants(self, max_pow):
        funcs = []
        for i in range(len(self.avaliable_funcs)):
            for j in range(1, max_pow+1):
                funcs.append((self.avaliable_funcs[i], j))
        return funcs