import numpy as np
import pandas as pd
import os
import scipy.optimize as opt


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def cost(theta, X, y):
    theta = np.matrix(theta)
    x_theta = X * theta.T
    h_x = sigmoid(x_theta)
    part1 = np.multiply(-y, np.log(h_x))
    part2 = np.multiply((1 - y), np.log(1 - h_x))
    return np.sum(part1 - part2) / (len(X))


def gradient(theta, X, y):
    theta = np.matrix(theta)
    x_theta = X * theta.T
    h_x = sigmoid(x_theta)
    return (h_x - y).T * X / len(X)


def predict(theta, X):
    probability = sigmoid(X * theta.T)
    return [1 if x >= 0.5 else 0 for x in probability]


def gradient_decent(theta, X, y, alpha=1e-7, max_iterations=1e4):
    tolerance = 1e-7
    cost_val = cost(theta, X, y)
    difference = tolerance + 1
    iteration = 0
    while (difference > tolerance) and (iteration < max_iterations):
        theta = theta + alpha * gradient(theta, X, y)
        temp = cost(theta, X, y)
        difference = np.abs(temp - cost_val)
        cost_val = temp
        iteration += 1
    return theta

if __name__ == "__main__":

    data = pd.read_csv(r'../data/ex2data1.txt', header=None, names=['Exam 1', 'Exam 2', 'Admitted'])
    data.insert(0, 'Ones', 1)

    cols = data.shape[1]
    X = data.iloc[:,0:cols-1]
    y = data.iloc[:,cols-1:cols]

    X = np.matrix(np.array(X.values))
    y = np.matrix(np.array(y.values))

    n = X.shape[1]

    theta = np.zeros(n)
    print cost(theta, X, y)

    result = opt.fmin_tnc(func=cost, x0=theta, fprime=gradient, args=(X, y))
    theta_min = np.matrix(result[0])
    print result

    predictions = predict(theta_min, X)
    correct = [1 if ((a == 1 and b == 1) or (a == 0 and b == 0)) else 0 for (a, b) in zip(predictions, y)]
    accuracy = (sum(map(int, correct)) % len(correct))

    print 'accuracy = {0}%'.format(accuracy)

    theta_min = gradient_decent(theta, X, y)
    predictions = predict(theta_min, X)
    correct = [1 if ((a == 1 and b == 1) or (a == 0 and b == 0)) else 0 for (a, b) in zip(predictions, y)]
    accuracy = (sum(map(int, correct)) % len(correct))

    print 'accuracy = {0}%'.format(accuracy)

