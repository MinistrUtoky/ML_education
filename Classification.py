import math

import numpy as np
from matplotlib import pyplot as plt


class Classification:
    N = 1000
    threshold = 0.5
    learning_rate = 1e-4

    def __init__(self):
        foot = self.random_normal_football(160, 10, self.N)
        basket = self.random_normal_basketball(165, 15, self.N)
        self.classify(foot, basket)

    #region Permutation
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
        return (np.array([Xtrain, Ytrain.reshape(1, Ytrain.shape[0])]),
                np.array([Xvalidation, Yvalidation.reshape(1, Yvalidation.shape[0])]),
                np.array([Xtest, Ytest.reshape(1, Ytest.shape[0])]))
    def monte_carlo_2(self, X, Y, train_slice=0.6, test_slice=0.2):
        return self.monte_carlo(
            np.concatenate(
                (X.copy().reshape(len(X), -1), Y.copy().reshape(len(Y), -1)),
                axis=1),
            train_slice, test_slice)
    def XY_train_validation_test(self, X, Y):
        train_percentage = 80 #random.Random.randint(0, 100)
        validation_percentage = 0 #random.Random.randint(0, 100 - train_percentage)
        test_percentage = 100-train_percentage-validation_percentage

        XY = self.monte_carlo_2(X, Y, train_percentage/100, test_percentage/100)
        XY_train = XY[0]
        XY_validation = XY[1]
        XY_test = XY[2]
        return XY_train, XY_validation, XY_test
    #endregion

    #region Classification
    def classify(self, footballers, basketballers):
        Y = [0]*len(footballers)
        X = np.append(footballers, basketballers)
        Y.extend([1]*len(basketballers))
        Y = np.array(Y)

        iters = 100

        XY_train, XY_validation, XY_test = self.XY_train_validation_test(X, Y)
        Xtrain, Ytrain = XY_train[0].T, XY_train[1].T
        Xtest, Ytest = XY_test[0].T, XY_test[1].T

        X_train_standardized = (Xtrain - Xtrain.mean()) / Xtrain.std()
        X_test_standardized = (Xtest - Xtest.mean()) / Xtest.std()

        weight = self.regress(X_train_standardized, Ytrain, self.learning_rate, iters)
        p = self.predict_classes(X_test_standardized, weight) > self.threshold
        TP, TN, FP, FN = 0, 0, 0, 0
        for i in range(len(p)):
            if Ytrain[i][0] == 1 and p[i] == 1:
                TP+=1
            elif Ytrain[i][0] == 0 and p[i] == 0:
                TN+=1
            elif Ytrain[i][0] == 0 and p[i] == 1:
                FP+=1
            elif Ytrain[i][0] == 1 and p[i] == 0:
                FN+=1

        print("For threshold of {0}:".format(self.threshold))
        self.write_metrics_from(TP, TN, FP, FN, len(X_train_standardized))

        self.brute_force_ROC(X_test_standardized, Ytest, weight)
        #self.ROC_and_stone(X_train_standardized, Ytrain, iters)

    def write_metrics_from(self, TP, TN, FP, FN, N):
        Accuracy = self.Accuracy(TP, TN, N)
        Precision = self.Precision(TP, FP)
        Recall = self.Recall(TP, TN)
        F1 = self.F1_score(Precision, Recall)
        alpha = self.alpha_error(FP, TN)
        beta = self.beta_error(FN, TP)
        print("TP:", TP)
        print("TN:", TN)
        print("FP:", FP)
        print("FN:", FN)
        print("Accuracy:", Accuracy)
        print("Precision:", Precision)
        print("Recall:", Recall)
        print("F1:", F1)
        print("alpha:", alpha)
        print("beta:", beta)

    def brute_force_ROC(self, X, Y, weight):
        thresholds = np.linspace(0,1,100)
        FTPRs = []
        Max_Accuracy, Max_TP_TN_FP_FN = 0, (0,0,0,0)
        for threshold in thresholds:
            self.threshold = threshold
            p = self.predict_classes(X, weight) > self.threshold
            TP, TN, FP, FN = 0, 0, 0, 0
            for i in range(len(p)):
                if Y[i][0] == 1 and p[i] == 1:
                    TP += 1
                elif Y[i][0] == 0 and p[i] == 0:
                    TN += 1
                elif Y[i][0] == 0 and p[i] == 1:
                    FP += 1
                elif Y[i][0] == 1 and p[i] == 0:
                    FN += 1
            Accuracy = self.Accuracy(TP, TN, len(X))
            if Accuracy > Max_Accuracy:
                Max_Accuracy = Accuracy
                Max_TP_TN_FP_FN = TP, TN, FP, FN
            FPR = FP / (FP + TN)
            TPR = TP / (TP + FN)
            FTPRs.append([FPR, TPR])
        FTPRs = np.array(FTPRs)
        FTPRs = FTPRs.T
        AUC = np.sum(np.trapz(FTPRs[0], FTPRs[1])) + 1
        xplt = FTPRs[0]
        yplt = FTPRs[1]
        TP, TN, FP, FN = Max_TP_TN_FP_FN
        print("\nAUC:{0}\n".format(AUC))
        print("For Max Accuracy")
        self.write_metrics_from(TP, TN, FP, FN, len(X))
        plt.plot([0, 1], [0, 1])
        plt.plot(xplt, yplt)
        plt.xlabel('FPR')
        plt.ylabel('TPR')
        plt.title('AUC={}'.format(round(AUC,4)))
        plt.show()

    def regress(self, Xtrain, Ytrain, learning_rate, iters):
        Xtrain = np.insert(Xtrain, 0, 1, axis=1)
        w = np.ones(Xtrain.shape[1])
        for _ in range(iters):
            w = w - learning_rate * self.gradient(Ytrain, Xtrain, w)
        return w

    def predict_classes(self, X, w):
        return self.y(np.insert(X, 0, 1, axis=1), w)
    #endregion

    #region Classification values
    @staticmethod
    def Accuracy(TP, TN, N):
        return (TP+TN)/N

    @staticmethod
    def Precision(TP, FP):
        return TP/(TP+FP)

    @staticmethod
    def Recall(TP, FN):
        return TP/(TP+FN)

    @staticmethod
    def F1_score(Precision, Recall):
        return 2*(Precision*Recall)/(Precision+Recall)

    @staticmethod
    def alpha_error(FP, TN):
        return FP/(TN+FP)

    @staticmethod
    def beta_error(FN, TP):
        return FN/(TP+FN)

    @staticmethod
    def FPR(FP, TN):
        return FP / (FP + TN)

    @staticmethod
    def TPR(TP, FN):
        return TP / (TP + FN)
    #endregion

    #region Named functions
    @staticmethod
    def g(a):
        return 1 / (1 + np.exp(-a))

    def y(self, F, w, b=0):
        return self.g(F@w)

    @staticmethod
    def w(lamda, F, t):
        return np.linalg.pinv(F.T @ F + lamda * np.identity(F.shape[1])) @ F.T @ t

    @staticmethod
    def E(w, x, t, lamda, y):
        return - sum(t*np.log(y) + (1-t)*np.log(1-y))

    def gradient(self, t, X, w):
        y_pred = self.y(X,w)
        return (y_pred-t.reshape(y_pred.shape)) @ X
    #endregion

    #region Generators
    @staticmethod
    def random_normal_football(mu_0, sigma_0, N):
        return np.random.normal(mu_0, sigma_0, (1, N))[0]

    @staticmethod
    def random_normal_basketball(mu_1, sigma_1, N):
        return np.random.normal(mu_1, sigma_1, (1, N))[0]
    #endregion

