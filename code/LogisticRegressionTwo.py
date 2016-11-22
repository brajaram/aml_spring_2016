import numpy as np
import pandas as pd
import os
import scipy.optimize as opt


def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def cost(theta, X, y):
    theta = np.matrix(theta)
    X = np.matrix(X)
    y = np.matrix(y)
    first = np.multiply(-y, np.log(sigmoid(X * theta.T)))
    second = np.multiply((1 - y), np.log(1 - sigmoid(X * theta.T)))
    return np.sum(first - second) / (len(X))

def gradient(theta, X, y):
    theta = np.matrix(theta)
    X = np.matrix(X)
    y = np.matrix(y)

    parameters = int(theta.ravel().shape[1])
    grad = np.zeros(parameters)

    error = sigmoid(X * theta.T) - y

    for i in range(parameters):
        term = np.multiply(error, X[:,i])
        grad[i] = np.sum(term) / len(X)

    return grad

def predict(theta, X):
    probability = sigmoid(X * theta.T)
    return [1 if x >= 0.5 else 0 for x in probability]


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
    print cost(theta, X, y)

    result = opt.fmin_tnc(func=cost, x0=theta, fprime=gradient, args=(X, y))
    theta_min = np.matrix(result[0])
    print theta_min
    predictions = predict(theta_min, X)
    correct = [1 if ((a == 1 and b == 1) or (a == 0 and b == 0)) else 0 for (a, b) in zip(predictions, y)]
    accuracy = (sum(map(int, correct)) % len(correct))
    print 'accuracy = {0}%'.format(accuracy)

