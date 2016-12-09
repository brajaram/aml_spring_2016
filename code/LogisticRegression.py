import numpy as np
import pandas as pd
import sys
import scipy.optimize as opt
import matplotlib.pyplot as plt
import itertools


def load_csv(file_name):
    """

    :param file_name:
    :return:
    """
    df = pd.read_csv(file_name)
    no_of_columns = len(df.columns)
    print '\nNumber of columns: ', no_of_columns
    for idx, name in enumerate(df.columns, start=1):
        if idx != no_of_columns:
            df[name] = df[name].astype(float)
    return df


def split_dataset(dataset, split_ratio):
    """

    :param dataset:
    :param split_ratio:
    :return:
    """
    msk = np.random.rand(len(dataset)) < split_ratio
    train_set = dataset[msk]
    test_set = dataset[~msk]
    return train_set, test_set


def sigmoid(z):
    """

    :param z:
    :return:
    """
    return 1 / (1 + np.exp(-z))


def cost(theta, X, y):
    """

    :param theta:
    :param X:
    :param y:
    :return:
    """
    theta = np.matrix(theta)
    x_theta = X * theta.T
    h_x = sigmoid(x_theta)
    part1 = np.multiply(-y, np.log(h_x))
    part2 = np.multiply((1 - y), np.log(1 - h_x))
    return np.sum(part1 - part2) / (len(X))


def gradient(theta, X, y):
    """

    :param theta:
    :param X:
    :param y:
    :return:
    """
    theta = np.matrix(theta)
    x_theta = X * theta.T
    h_x = sigmoid(x_theta)
    return (h_x - y).T * X / len(X)


def predict(theta, X):
    """

    :param theta:
    :param X:
    :return:
    """
    probability = sigmoid(X * theta.T)
    return [1 if x >= 0.5 else 0 for x in probability]


def gradient_decent(theta, X, y, alpha=1e-7, max_iterations=1e4):
    """

    :param theta:
    :param X:
    :param y:
    :param alpha:
    :param max_iterations:
    :return:
    """
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


def get_accuracy(test_set, predictions):
    """

    :param test_set:
    :param predictions:
    :return:
    """
    correct = 0
    for x in xrange(len(test_set)):
        if test_set[x] == predictions[x]:
            correct += 1
    return (correct/float(len(test_set))) * 100.0


# calculate a confusion matrix
def confusion_matrix(actual, predicted):
    """

    :param actual:
    :param predicted:
    :return:
    """
    actuals = []
    for x in range(len(actual)):
        actuals.append(actual[x])
    unique = set(actuals)

    matrix = [list() for x in range(len(unique))]
    for i in range(len(unique)):
        matrix[i] = [0 for x in range(len(unique))]
    lookup = dict()
    for i, value in enumerate(unique):
        lookup[value] = i

    for i in range(len(actuals)):
        x = lookup[actuals[i]]
        y = lookup[predicted[i]]
        matrix[x][y] += 1
    return unique, matrix


def print_confusion_matrix(unique, matrix):
    """

    :param unique:
    :param matrix:
    :return:
    """
    print('\n')
    header = '   ' + ' | '.join(str(x).ljust(5,' ') for x in unique) + '|'
    header_len = len(header)

    print(header + '<--- predicted')
    print('-' * header_len)
    for i, x in enumerate(unique):
        print("%s| %s|" % (x, ' | '.join(str(x).ljust(5,' ') for x in matrix[i])))
        print('-' * header_len)


def plot_confusion_matrix(cm, classes,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()


def convert_class_to_numeric(data, last_col_idx):
    """

    :param data:
    :param last_col_idx:
    :return:
    """
    unique_classes = np.sort(data.iloc[:,last_col_idx].unique())
    class_dict = {}
    for idx,class_nm in enumerate(unique_classes,start=1):
        class_dict[class_nm] = idx
    data.iloc[:, last_col_idx] = data.iloc[:,last_col_idx].map(class_dict)
    return data


def get_x_y(data,cols):
    """

    :param data:
    :param cols:
    :return:
    """
    X = data.iloc[:, 0:cols - 1]
    y = data.iloc[:, cols-1:cols]
    X = np.matrix(np.array(X.values))
    y = np.matrix(np.array(y.values))
    return X, y


def main(file_name, conv_flag):
    """

    :param file_name:
    :param conv_flag:
    :return:
    """
    split_ratio = 0.70
    data = load_csv(file_name)
    last_col_idx = len(data.columns) - 1
    if conv_flag == 'yes':
        data = convert_class_to_numeric(data, last_col_idx)
    data.insert(0, 'Ones', 1)
    cols = data.shape[1]
    training_set, test_set = split_dataset(data, split_ratio)
    print('\nSplit {0} rows into train={1} and test={2} rows').format(len(data), len(training_set), len(test_set))
    train_X, train_y = get_x_y(training_set,cols)
    test_X, test_y = get_x_y(test_set,cols)
    n = train_X.shape[1]
    theta = np.zeros(n)
    result = opt.fmin_tnc(func=cost, x0=theta, fprime=gradient, args=(train_X, train_y))
    theta_min = np.matrix(result[0])
    predictions = predict(theta_min, test_X)
    accuracy = get_accuracy(test_y, predictions)
    print '\naccuracy = {0}%'.format(accuracy)
    # # # Confusion Matrix
    unique, matrix = confusion_matrix(test_y.ravel().tolist()[0], predictions)
    plot_confusion_matrix(np.array(matrix), unique)
    print('\n')


if __name__ == "__main__":

    file_name = sys.argv[1]
    convert_class = sys.argv[2]
    main(file_name,convert_class)