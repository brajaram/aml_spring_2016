import numpy as np
import pandas as pd
import os
import scipy.optimize as opt



class LogisiticRegression:

    def __init__(self, X, y, tolerance=1e-5):
        """

        :param theta:
        :param X:
        :param y:
        """
        self.tolerance = tolerance
        self.X = np.matrix(X)
        self.y = np.matrix(y)
        self.theta = np.zeros(X.shape[1])
        self.likelihood_history = []
        self.m = len(y)

    def cost(self):
        self.theta = np.matrix(self.theta)
        first = np.multiply(-self.y, np.log(self.sigmoid(self.X * self.theta.T)))
        second = np.multiply((1 - self.y), np.log(1 - self.sigmoid(self.X * self.theta.T)))
        return np.sum(first - second) / (len(X))

    def sigmoid(self,z):
        return 1 / (1 + np.exp(-z))

    def gradient(self):
        theta = np.matrix(self.theta)
        parameters = int(theta.ravel().shape[1])
        grad = np.zeros(parameters)
        error = self.sigmoid(self.X * theta.T) - self.y
        for i in range(parameters):
            term = np.multiply(error, self.X[:, i])
            grad[i] = np.sum(term) / len(self.X)
        print grad
        return np.matrix(grad)

    def gradient_decent(self, alpha=0.0001, max_iterations=10):
        """Runs the gradient decent algorithm

        Parameters
        ----------
        alpha : float
            The learning rate for the algorithm

        max_iterations : int
            The maximum number of iterations allowed to run before the algorithm terminates

        """
        previous_likelihood = self.cost()
        difference = self.tolerance + 1
        converged = False
        iteration = 0
        while not converged:
            self.theta = self.theta - (self.gradient() * alpha)
            temp = self.cost()
            difference = np.abs(temp - previous_likelihood)
            previous_likelihood = temp
            self.likelihood_history.append(previous_likelihood)
            iteration += 1


if __name__ == "__main__":

    data = pd.read_csv(r'../data/ex2data1.txt', header=None, names=['Exam 1', 'Exam 2', 'Admitted'])
    data.insert(0, 'Ones', 1)

    cols = data.shape[1]
    X = data.iloc[:,0:cols-1]
    y = data.iloc[:,cols-1:cols]

    X = np.array(X.values)
    y = np.array(y.values)

    n = X.shape[1]

    theta = np.zeros(n)

    lg = LogisiticRegression(X,y)
    lg.gradient_decent()
    print lg.theta
    print lg.cost()

