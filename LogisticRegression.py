import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

class LogisticRegression:
    def __init__(self):
        self.X_train = []
        self.y_train = []
        self.thetas = []
        self.thetaXs = []

    def run(self):
        col_names = ['Survival', 'PassClass', 'Sex', 'Age', 'SiblingsOrSpouse', 'ParentsOrKids', 'TicketCost']
        table = pd.read_excel("Data/BD_Titanik.xlsx", header=None, names=col_names, skiprows=3)
        dataframes = [x for _, x in table.groupby('Survival')]
        df0 = dataframes[1].reset_index(drop=True)
        df1 = dataframes[0].drop(columns=["Survival"]).reset_index(drop=True)
        df0["Survival"] = 0
        df1["Survival"] = 1
        train_df0 = df0.iloc[: int(len(df0)*1)]
        train_df1 = df1.iloc[: int(len(df1)*1)]
        self.X_train = pd.concat([train_df0.drop(columns=["Survival"]),
                             train_df1.drop(columns=["Survival"])]).reset_index(drop=True)
        self.y_train = pd.concat([train_df0["Survival"],
                             train_df1["Survival"]]).to_numpy()
        self.mean_train = self.X_train.mean()
        self.std_train = self.X_train.std()
        X_train_standardized = ((self.X_train - self.mean_train) / self.std_train).to_numpy()
        self.X_train = np.hstack([np.ones((X_train_standardized.shape[0], 1)), X_train_standardized])
        theta = self.training_ground(self.X_train, self.y_train, epochs=50, learning_rate=0.01)
        print(f"Theta^: {theta}")
        print(f"l(theta^): {self.MLE(theta)}")
        self.build_plot(100, 0.01)
        #---------------------------------------------------------
        '''
        test_df0 = df0.iloc[int(len(df0) * 0.9995):]
        test_df1 = df1.iloc[int(len(df1) * 0.9995):]
        X_test = pd.concat([test_df0.drop(columns=["Survival"]), test_df1.drop(columns=["Survival"])]).reset_index(drop=True)
        X_test_standard = ((X_test - self.mean_train) / self.std_train).to_numpy()
        X_test = np.hstack([np.ones((X_test_standard.shape[0], 1)), X_test_standard])'''
        my_data = [1, 1, 33, 0, 0, 50]
        print("Your data: " + str(my_data))
        X_test = [np.insert(((my_data - self.mean_train) / self.std_train), 0, 1)]
        print(self.probability(theta, X_test))
        if self.predict(self.probability(theta, X_test))[0] == 1:
            print("You'd survive\n")
        else:
            print("You'd die for sure\n")# а если поставить цену билета 100 то умираю ахах
        self.w_distribution_and_confidence_interval(theta)

    def sigmoid_chance_to_live(self, thetaT_x):
        return 1 / (1 + np.exp(thetaT_x))

    def probability(self, weight, X):
        y_probability = np.array([])
        for i in range(len(X)):
            y_probability = np.append(y_probability, self.sigmoid_chance_to_live(-np.dot(weight, X[i])))
        return y_probability

    def gradient(self, X_input, real_y, y_probability):
        return np.dot((real_y - y_probability), X_input)

    def training_ground(self, X, y, epochs=30, learning_rate=0.01):
        theta = np.ones((1, self.X_train.shape[1]))
        for k in range(epochs):
            theta = theta + learning_rate * self.gradient(X, y, self.probability(theta, X))
        return theta

    def predict(self, y_probability):
        y_prediction = np.array([], dtype=np.int16)
        for i in range(len(y_probability)):
            if y_probability[i] > 0.5:
                prediction = 1
            else:
                prediction = 0
            y_prediction = np.append(y_prediction, prediction)
        return y_prediction

    def MLE(self, theta):
        l_theta = 0
        for i in range(len(self.X_train)):
            l_theta += np.log(self.sigmoid_chance_to_live(-np.dot(theta, self.X_train[i]))**self.y_train[i]
                              * self.sigmoid_chance_to_live(np.dot(theta, self.X_train[i]))**(1-self.y_train[i]))
        return l_theta

    def build_plot(self, until, learning_rate):
        theta = np.ones((1, self.X_train.shape[1]))
        for k in range(until):
            theta = theta + learning_rate * self.gradient(self.X_train, self.y_train, self.probability(theta, self.X_train))
        mean_train = np.zeros(7)
        for i in range(len(self.X_train)):
            mean_train += self.X_train[i]
        mean_train /= len(self.X_train)
        mean_train = mean_train[1:]
        std_train = 0
        for i in range(len(self.X_train)):
            std_train += np.exp(-mean_train*self.X_train[i][1:].T)/(1 + np.exp(-mean_train*self.X_train[i][1:].T))**2 * self.X_train[i][1:]*self.X_train[i][1:].T
        std_train = 1 - std_train**-1
        x = np.linspace(mean_train - 3*std_train, mean_train + 3*std_train, len(self.X_train))
        plt.plot(x, stats.norm.pdf(x, mean_train, std_train))
        plt.show()

    def w_distribution_and_confidence_interval(self, theta):
        theta = theta[0][1:]
        x_test = [np.insert((([3, 0, 33, 1, 0, 70] - self.mean_train) / self.std_train), 0, 1)][0][1:]
        self.X_train = self.X_train[:, 1:]
        std_w = 0
        for i in range(len(self.X_train)):
            self.X_train[i] = np.dot(theta, self.X_train[i])
            std_w += self.X_train[i]**2/len(self.X_train)
        std_w = std_w**0.5
        print(stats.t.interval(alpha=0.1, df=len(self.X_train)-1, loc=0, scale=std_w[0]))
        print(np.dot(theta, x_test))
