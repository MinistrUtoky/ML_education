import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_digits

'''
Бонусные задания (необязательные):
3. Произвести валидацию гиперпараметров с помощью рандомизированной процедуры:
	- максимальная глубина деревьев
	- критерий (энтропия/неопределенности Джини/ошибку классификации)
	- минимальное значение критерия
	- минимальное количество элементов в терминальных узлах

После валидации вычислить точность и confusion matrix на лучших параметрах.

4. Реализовать вычисление оптимальных параметров разделяющей функции на основе ограниченного перебора этих параметров.

5. В качестве разделяющих функций использовать разделение гиперплоскостями не параллельными осям координат.

6. В качестве разделяющих функций использовать разделение гиперповерхностями, 
полученными с помощью базисных функций phi (нелинейное разделение).
'''

class BinaryDecisionTree:
    H_min = 0.1
    N_min = 1
    Depth_max = 20
    depth_array = [[] for _ in range(Depth_max+1)]
    node0 = [None, None, None]

    def __init__(self):
        X, Y, classes = self.fetch_data()
        XY_train, XY_validation, XY_test = self.XY_train_validation_test(X, Y)
        Xtrain, Ytrain = XY_train[0].T, XY_train[1].T
        Xtest, Ytest = XY_test[0].T, XY_test[1].T
        self.train(Xtrain, Ytrain, classes)
        print("Train set reivew:")
        self.test_and_review(Xtrain, Ytrain, classes)
        print("Test set reivew:")
        self.test_and_review(Xtest, Ytest, classes)

    def fetch_data(self):
        digits = load_digits()
        X = digits.data
        Y = digits.target
        classes = digits.target_names
        return X, Y, classes

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
        train_percentage = 80 #random.Random.randint(0, 100)
        validation_percentage = 0 #random.Random.randint(0, 100 - train_percentage)
        test_percentage = 20#100-train_percentage-validation_percentage

        XY = self.monte_carlo_2(X, Y, train_percentage/100, test_percentage/100)
        XY_train = XY[0]
        XY_validation = XY[1]
        XY_test = XY[2]
        return XY_train, XY_validation, XY_test
    #endregion

    #region Training
    def train(self, Xtrain, Ytrain, classes):
        indices = True
        x = Xtrain.T.copy()
        y = Ytrain.copy()
        attribute_index = 0
        xi, yi = x[attribute_index][indices][0], y[indices][0].reshape(1, -1)[0]
        H = self.enthropy(y, classes)
        # choosing tau
        best_tau = self.best_tau_for_a_node(xi, yi, H, classes)
        # halve sample with given tau
        self.node0 = [best_tau, self.collect_child_nodes(x, y, attribute_index+1, xi<best_tau, xi>best_tau, classes, 1), yi[indices]]
        #self.show_tree(self.node0)
        #self.show_tree_ends(self.node0, 0)

    def collect_child_nodes(self, X, Y, attribute_index, left_indices, right_indices, classes, depth):
        node_left, node_right = None, None
        if attribute_index>63 or depth == self.Depth_max:
            return (node_left, node_right)
        #left
        xi, yi = X[attribute_index], Y.reshape(1, -1)[0]
        if len(xi[left_indices]) == self.N_min:
            node_left = [None, None, yi[left_indices]]
        elif len(xi[left_indices]) > self.N_min and sum(left_indices) != 0:
            H = self.enthropy(yi[left_indices], classes)
            if H < self.H_min:
                node_left = [None, None, yi[left_indices]]
            else:
                best_tau = self.best_tau_for_a_node(xi[left_indices], yi[left_indices], H, classes)
                node_left = [best_tau, self.collect_child_nodes(X, Y, attribute_index+1,
                                                np.logical_and(left_indices, xi<best_tau),
                                                np.logical_and(left_indices, xi>best_tau), classes, depth+1), yi[left_indices]]
        #right
        xi, yi = X[attribute_index], Y.reshape(1, -1)[0]
        if len(xi[right_indices]) == self.N_min:
            node_right = [None, None, yi[right_indices]]
        elif len(xi[right_indices]) > self.N_min and sum(right_indices) != 0:
            H = self.enthropy(yi[right_indices], classes)
            if H < self.H_min:
                node_right = [None, None, yi[right_indices]]
            else:
                best_tau = self.best_tau_for_a_node(xi[right_indices], yi[right_indices], H, classes)
                node_right = [best_tau, self.collect_child_nodes(X, Y, attribute_index+1,
                                                 np.logical_and(right_indices, xi < best_tau),
                                                 np.logical_and(right_indices, xi > best_tau), classes, depth+1), yi[right_indices]]
        return (node_left, node_right)

    def best_tau_for_a_node(self, X, Y, H, classes):
        Taus = np.unique(X) + 0.5
        max_Info_gain = 0
        best_tau = Taus[0]
        for tau in Taus:
            I = self.binary_information_gain(X, Y, H, tau, classes)
            if I > max_Info_gain:
                max_Info_gain = I
                best_tau = tau
        return best_tau

    def show_tree_ends(self, node, depth):
        if node!=None:
            if node[1] == None:
                print(depth, node[0], node[2])
            else:
                self.show_tree_ends(node[1][0], depth+1)
                self.show_tree_ends(node[1][1], depth+1)

    def show_tree(self, node0):
        def go_down_the_tree(node, depth):
            if node != None:
                self.depth_array[depth].append(str(node[0]))
                if node[1] != None:
                    go_down_the_tree(node[1][0], depth+1)
                    go_down_the_tree(node[1][1], depth+1)
            else:
                self.depth_array[depth].append("None")
        go_down_the_tree(node0, 0)
        padding_coef = len(max(self.depth_array, key=len)) * 2
        for i in range(len(self.depth_array)):
            print(" " * padding_coef + str(self.depth_array[i]))
            padding_coef -= 2
    #endregion

    def test_and_review(self, X, Y, classes):
        p, pbs = self.predict_classes(X)
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
        print("Accuracy:", Accuracy)
        print("Confusion matrix:")
        print(confusion_matrix)
        self.hist_probabilites(wrong_pbs, right_pbs)

    def hist_probabilites(self, wrong_pbs, right_pbs):
        plt.hist(wrong_pbs, color='r', label='Wrong decisions confidence')
        plt.xlabel("set index")
        plt.ylabel("probability")
        plt.xlim((0, 1))
        plt.legend()
        plt.show()
        plt.hist(right_pbs, color='b', label='Right decisions confidence')
        plt.xlabel("set index")
        plt.ylabel("probability")
        plt.xlim((0, 1))
        plt.legend()
        plt.show()

    def predict_classes(self, X):
        predictions, probabilities = [], []
        for x in X:
            prediction, probability = self.predict_class(x)
            predictions.append(prediction)
            probabilities.append(probability)
        return predictions, probabilities

    def predict_class(self, x):
        '''
        :param x: set of attributes
        :return: tuple (prediction, probability)
        '''
        current_node = self.node0
        i = 0
        while current_node[0] != None:
            if x[i] > current_node[0]:
                if current_node[1][1] == None:
                    break
                current_node = current_node[1][1]
            else:
                if current_node[1][0] == None:
                    break
                current_node = current_node[1][0]
            i+=1
        node_result = current_node[2].tolist()
        prediction = max(node_result, key=node_result.count)
        probability = node_result.count(prediction)/len(node_result)
        return prediction, probability



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
        Hs = np.array([sum(X < tau) / len(Y) * self.enthropy(Y[X < tau], classes),
              sum(X > tau) / len(Y) * self.enthropy(Y[X > tau], classes)])
        I = H - sum(np.array(Hs)) # information gain
        return I
    #endregion

