import random

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_digits

class BinaryDecisionTree:
    H_min = 0.001
    N_min = 1
    Depth_max = 16

    def __init__(self):
        X, Y, classes = self.fetch_data()
        XY_train, XY_validation, XY_test = self.XY_train_validation_test(X, Y)
        Xtrain, Ytrain = XY_train[0].T, XY_train[1].T
        Xvalidation, Yvalidation = XY_validation[0].T, XY_validation[1].T
        Xtest, Ytest = XY_test[0].T, XY_test[1].T
        '''
        feature_indices = np.arange(Xtrain.shape[1])
        feature_indices = np.random.permutation(feature_indices).reshape(int(Xtrain.shape[1]/2), 2)
        ABs = np.random.normal(size=Xtrain.shape[1]).reshape(int(Xtrain.shape[1]/2), 2)
        Xtrainhyperplaned = self.hyperplane_it(Xtrain, feature_indices, ABs)
        Xtesthyperplaned = self.hyperplane_it(Xtest, feature_indices, ABs)

        possible_bases = [np.cos, np.sin, np.tan, np.tanh, np.cosh, np.sinh, np.round]
        bases = [possible_bases[random.randint(0, len(possible_bases)-1)] for i in range(int(Xtrain.shape[1]/2))]
        Xtrainhypersurfaced = self.hypersurface_it(Xtrain, feature_indices, ABs, bases)
        Xtesthypersurfaced = self.hypersurface_it(Xtest, feature_indices, ABs, bases)'''

        node0 = self.train_with_validation(Xtrain, Ytrain, Xvalidation, Yvalidation, classes)#self.train_with_validation(Xtrain, Ytrain, classes)
        print("Train set reivew:")
        Accuracy, confusion_matrix, wrong_pbs, right_pbs = self.test_and_review(node0, Xtrain, Ytrain, classes)
        print("Accuracy:", Accuracy)
        print("Confusion matrix:")
        print(confusion_matrix)
        self.hist_probabilites(wrong_pbs, right_pbs)
        print("Test set reivew:")
        Accuracy, confusion_matrix, wrong_pbs, right_pbs = self.test_and_review(node0, Xtest, Ytest, classes)
        print("Accuracy:", Accuracy)
        print("Confusion matrix:")
        print(confusion_matrix)
        self.hist_probabilites(wrong_pbs, right_pbs)


    #region Data
    def fetch_data(self):
        digits = load_digits()
        X = digits.data
        Y = digits.target
        classes = digits.target_names
        return X, Y, classes

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
        return [[Xtrain, Ytrain.reshape(1, Ytrain.shape[0])],
                [Xvalidation, Yvalidation.reshape(1, Yvalidation.shape[0])],
                [Xtest, Ytest.reshape(1, Ytest.shape[0])]]

    def monte_carlo_2(self, X, Y, train_slice=0.6, test_slice=0.2):
        return self.monte_carlo(
            np.concatenate(
                (X.copy().reshape(len(X), -1), Y.copy().reshape(len(Y), -1)),
                axis=1),
            train_slice, test_slice)

    def XY_train_validation_test(self, X, Y):
        train_percentage = 40#random.Random.randint(0, 100)
        validation_percentage = 40 #random.Random.randint(0, 100 - train_percentage)
        test_percentage = 10#100-train_percentage-validation_percentage

        XY = self.monte_carlo_2(X, Y, train_percentage/100, test_percentage/100)
        XY_train = XY[0]
        XY_validation = XY[1]
        XY_test = XY[2]
        return XY_train, XY_validation, XY_test

    def hyperplane_it(self, X, feature_indices, abs):
        X = X.T.copy()
        for i in range(len(feature_indices)):
            X[feature_indices[i][0]] = abs[i][0]*X[feature_indices[i][0]] + abs[i][1]*X[feature_indices[i][1]]
        return X.T

    def hypersurface_it(self, X, feature_indices, abs, bases):
        X = X.T.copy()
        for i in range(len(feature_indices)):
            X[feature_indices[i][0]] = abs[i][0]*bases[i](X[feature_indices[i][0]]) + abs[i][1]*bases[i](X[feature_indices[i][1]])
        return X.T
    #endregion

    #region Training
    def train_with_validation(self, Xtrain, Ytrain, Xvalidation, Yvalidation, classes):
        def train(Xtrain, Ytrain, classes):
            coord_indices = True
            x = Xtrain.T.copy()
            y = Ytrain.copy()
            attribute_index = random.randint(0, Xtrain.shape[1]-1)
            xi, yi = x[attribute_index][coord_indices][0], y[coord_indices][0].reshape(1, -1)[0]
            H = self.criteria(y, classes)
            best_attribute_index, best_tau = self.best_tau_for_a_node(x, y, H, classes)
            node0 = [(attribute_index, best_tau),
                     self.collect_child_nodes(x, y, best_attribute_index, xi < best_tau, xi >= best_tau, classes, 1),
                     yi[coord_indices][0]]
            return node0
        training_iterations = 10
        max_acc = 0
        criterias = [self.enthropy, self.gini, self.misclassification_error]
        self.criteria = self.enthropy
        best_depth = 0
        best_criteria = self.enthropy
        best_criteria_delta = 1e+256
        best_N_min = 0
        best_node0 = [None, None, None]
        for i in range(training_iterations):
            print("Iteration", i)
            self.Depth_max = random.randint(8, 32)
            self.criteria = criterias[random.randint(0, 2)]
            self.H_min = random.randint(1, 1000) / 100000.0
            self.N_min = random.randint(1, 2)
            node0 = train(Xtrain, Ytrain, classes)
            acc, _, _, _ = self.test_and_review(node0, Xvalidation, Yvalidation, classes)
            if acc > max_acc:
                max_acc = acc
                best_criteria = self.criteria
                best_depth = self.Depth_max
                best_N_min = self.N_min
                best_criteria_delta = self.H_min
                best_node0 = node0
        print("Best depth:", best_depth)
        print("Best criteria:", best_criteria.__name__)
        print("Best terminal node size:", best_N_min)
        print("Best min criteria:", best_criteria_delta)
        return best_node0

    def collect_child_nodes(self, X, Y, attribute_index, left_indices, right_indices, classes, depth):
        node_left, node_right = None, None
        if depth == self.Depth_max:
            return (node_left, node_right)
        #left
        xi, yi = X[attribute_index], Y.reshape(1, -1)[0]
        if len(xi[left_indices]) <= self.N_min:
            node_left = [None, None, yi[left_indices]]
        elif len(xi[left_indices]) > self.N_min and sum(left_indices) != 0:
            H = self.criteria(yi[left_indices], classes)
            if H < self.H_min:
                node_left = [None, None, yi[left_indices]]
            else:
                best_attribute_index, best_tau = self.best_tau_for_a_node(X.T[left_indices].T, yi[left_indices], H, classes)
                if best_attribute_index is None:
                    return [None, None, yi[left_indices]]
                node_left = [(attribute_index, best_tau), self.collect_child_nodes(X, Y, best_attribute_index,
                                                np.logical_and(left_indices, xi<best_tau),
                                                np.logical_and(left_indices, xi>=best_tau), classes, depth+1), yi[left_indices]]
        #right
        xi, yi = X[attribute_index], Y.reshape(1, -1)[0]
        if len(xi[right_indices]) <= self.N_min:
            node_right = [None, None, yi[right_indices]]
        elif len(xi[right_indices]) > self.N_min and sum(right_indices) != 0:
            H = self.criteria(yi[right_indices], classes)
            if H < self.H_min:
                node_right = [None, None, yi[right_indices]]
            else:
                best_attribute_index, best_tau = self.best_tau_for_a_node(X.T[right_indices].T, yi[right_indices], H, classes)
                if best_attribute_index is None:
                    return [None, None, yi[left_indices]]
                node_right = [(attribute_index, best_tau), self.collect_child_nodes(X, Y, best_attribute_index,
                                                 np.logical_and(right_indices, xi < best_tau),
                                                 np.logical_and(right_indices, xi >= best_tau), classes, depth+1), yi[right_indices]]
        return (node_left, node_right)

    def best_tau_for_a_node(self, X, Y, H, classes):
        best_attribute_index = None
        best_tau = None
        max_Info_gain = 0
        for next_attr_idx in range(len(X)):
            Taus = np.unique(X[next_attr_idx])
            if len(Taus) > 1:
                for i in range(len(Taus)-1):
                    Taus[i] = (Taus[i] + Taus[i+1])/2.0
                Taus = Taus[:-1]
            else:
                Taus += 0.5
            for tau in Taus:
                I = self.binary_information_gain(X[next_attr_idx], Y, H, tau, classes)
                if I > max_Info_gain:
                    max_Info_gain = I
                    best_tau = tau
                    best_attribute_index = next_attr_idx
        return best_attribute_index, best_tau
    #endregion

    #region Test and review
    def test_and_review(self, node0, X, Y, classes):
        p, pbs = self.predict_classes(node0, X)
        correct_predictions = 0
        confusion_matrix = np.zeros(shape=(len(classes), len(classes)))
        wrong_pbs, right_pbs = [], []
        for i in range(len(p)):
            confusion_matrix[int(Y[i][0])][int(p[i])] += 1
            if Y[i][0] == p[i]:
                right_pbs.append(pbs[i])
                correct_predictions += 1
            else:
                wrong_pbs.append(pbs[i])
        Accuracy = correct_predictions/len(Y)
        return Accuracy, confusion_matrix, wrong_pbs, right_pbs

    def hist_probabilites(self, wrong_pbs, right_pbs):
        plt.hist(wrong_pbs, color='r', label='Wrong decisions confidence')
        plt.xlabel("set index")
        plt.ylabel("probability")
        plt.xlim((-0.1, 1.1))
        plt.legend()
        plt.show()
        plt.hist(right_pbs, color='b', label='Right decisions confidence')
        plt.xlabel("set index")
        plt.ylabel("probability")
        plt.xlim((-0.1, 1.1))
        plt.legend()
        plt.show()

    def predict_classes(self, node0, X):
        predictions, probabilities = [], []
        for x in X:
            pred = self.predict_class2(node0, x)
            if pred is not None:
                prediction, probability = pred
                predictions.append(prediction)
                probabilities.append(probability)
        return predictions, probabilities

    def predict_class2(self, node, X):
        if node is None:
            return
        if node[0] is None:
            if (len(node[2])==0):
                return None
            node_result = node[2].tolist()
            prediction = max(node_result, key=node_result.count)
            probability = node_result.count(prediction)/len(node_result)
            return prediction, probability
        x = X[node[0][0]]
        if x >= node[0][1]:
            pred = self.predict_class2(node[1][1], X)
            if pred is None:
                return None
            else:
                return pred
        else:
            pred = self.predict_class2(node[1][0], X)
            if pred is None:
                return None
            else:
                return pred

    @staticmethod
    def predict_class(node0, x):
        current_node = node0
        while True:
            if current_node[0] is None:
                break
            if current_node[0][0] is None:
                break
            if x[current_node[0][0]] >= current_node[0][1]:
                if len(current_node[1][1][2]) == 0:
                    break
                current_node = current_node[1][1]
            else:
                if len(current_node[1][0][2]) == 0:
                    break
                current_node = current_node[1][0]
        node_result = current_node[2].tolist()
        prediction = max(node_result, key=node_result.count)
        probability = node_result.count(prediction)/len(node_result)
        return prediction, probability
    #endregion

    #region Information
    @staticmethod
    def enthropy(Y : np.array, classes : list):
        '''
        :param Y: classes for a given sample
        :param classes: classes list
        :return: enthropy (H)
        '''
        N = Y.shape[0]
        Ni = [0]*len(classes)
        for i in range(len(classes)):
            Ni[classes[i]] = sum(Y==classes[i])
        Ni = np.array(Ni).reshape(1, -1)[0]
        Ni = Ni[Ni!=0]
        return -np.sum(Ni/N * np.log(Ni/N))

    @staticmethod
    def gini(Y : np.array, classes : list):
        '''
        :param Y: classes for a given sample
        :param classes: classes list
        :return: gini index (Gini)
        '''
        N = Y.shape[0]
        Ni = [0]*len(classes)
        for i in range(len(classes)):
            Ni[classes[i]] = sum(Y==classes[i])
        Ni = np.array(Ni).reshape(1, -1)[0]
        Ni = Ni[Ni!=0]
        return 1 - sum((Ni/N)**2)

    @staticmethod
    def misclassification_error(Y : np.array, classes : list):
        '''
        :param Y: classes for a given sample
        :param classes: classes list
        :return: gini index (Gini)
        '''
        N = Y.shape[0]
        Ni = [0]*len(classes)
        for i in range(len(classes)):
            Ni[classes[i]] = sum(Y==classes[i])
        Ni = np.array(Ni).reshape(1, -1)[0]
        Ni = Ni[Ni!=0]
        if len(Ni/N)==0:
            return 1
        return 1 - max(Ni/N)

    def binary_information_gain(self, X, Y, H, tau, classes):
        '''
        :param X: given part of a sample
        :param Y: part of classes as per X
        :param H: given node's enthropy
        :param tau: sample value threshold
        :param classes: classes list
        :return: information gain for a given binary decision tree's node
        '''
        Hs = np.array([sum(X < tau) / len(Y) * self.criteria(Y[X < tau], classes),
              sum(X >= tau) / len(Y) * self.criteria(Y[X >= tau], classes)])
        I = H - sum(np.array(Hs)) # information gain
        return I
    #endregion

