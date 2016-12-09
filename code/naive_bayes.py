import sys
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import itertools


def load_csv(filename):
    df = pd.read_csv(filename)
    #print '\nl type : ', type(df)
    no_of_columns = len(df.columns)
    print '\nNumber of columns: ', no_of_columns
    for idx, name in enumerate(df.columns, start=1):
        if idx != no_of_columns:
            df[name] = df[name].astype(float)
    return df


def split_dataset(dataset, split_ratio):
    msk = np.random.rand(len(dataset)) < split_ratio 
    train_set = dataset[msk]
    test_set = dataset[~msk]
    return train_set, test_set


def separate_by_class(dataset):
    separated = {}
    no_of_columns = len(dataset.columns) - 2
    class_col_nm = dataset.columns[-1]
    unique_classes = dataset[class_col_nm].unique()
    #initiate dictionary with classes as key
    for class_value in unique_classes:
        separated[class_value] = dataset[dataset[class_col_nm] == class_value].iloc[:,0:no_of_columns].values.tolist()
    return separated


def mean(numbers):
    return sum(numbers)/float(len(numbers))


def stdev(numbers):
    avg = mean(numbers)
    variance = sum([pow(x - avg,2) for x in numbers])/float(len(numbers) - 1)
    return math.sqrt(variance)


def summarize(dataset):
    #The zip function groups the values for each attribute across our data instances into their own lists
    summaries = [(mean(attribute), stdev(attribute)) for attribute in zip(*dataset)]
    del summaries[-1]
    return summaries


def summarize_by_class(dataset):
    separated = separate_by_class(dataset)
    summaries = {}
    for class_value, instances in separated.iteritems():
        print '\nclass_value    : ', str(class_value)
        print '\n# of instances : ', str(len(instances))
        summaries[class_value] = summarize(instances)
    return summaries


def calculate_probability(x, mean, stdev):
    if stdev > 0:
        exponent = math.exp(-(math.pow(x-mean,2)/(2*math.pow(stdev,2))))
    	output = (1 / (math.sqrt(2*math.pi) * stdev)) * exponent
    else:
        output = 0
    return output

def calculate_class_probabilities(summaries, input_vector):
    probabilities = {}
    for class_value, class_summaries in summaries.iteritems():
        probabilities[class_value] = 1
        for i in range(len(class_summaries)):
            mean, stdev = class_summaries[i]
            x = input_vector[i]
            probabilities[class_value] *= calculate_probability(x, mean, stdev)
    return probabilities


def predict(summaries, input_vector):
    probabilities = calculate_class_probabilities(summaries, input_vector)
    bestLabel, bestProb = None, -1
    for classValue, probability in probabilities.iteritems():
        if bestLabel is None or probability > bestProb:
            bestProb = probability
            bestLabel = classValue
    return bestLabel


def get_predictions(summaries, test_set):
    predictions = []
    for i in range(len(test_set)):
        result = predict(summaries, test_set[i])
        predictions.append(result)
    return predictions


def get_accuracy(test_set, predictions):
    correct = 0
    for x in xrange(len(test_set)):
        if test_set[x] == predictions[x]:
            correct += 1
    return (correct/float(len(test_set))) * 100.0


# calculate a confusion matrix
def confusion_matrix(actual, predicted):
    #print 'actual : ', actual
    #print 'predicted : ', predicted
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
        #lookup[i] = value
    #print 'lookup : ', lookup

    for i in range(len(actuals)):
        x = lookup[actuals[i]]
        #print 'x : ', str(x)
        y = lookup[predicted[i]]
        matrix[x][y] += 1
    return unique, matrix


# pretty print a confusion matrix
def print_confusion_matrix(unique, matrix):
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
    Normalization can be applied by setting `normalize=True`.
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


def main(filename):
    split_ratio = 0.70
    dataset = load_csv(filename)
    training_set, test_set = split_dataset(dataset, split_ratio)
    print('\nSplit {0} rows into train={1} and test={2} rows').format(len(dataset), len(training_set), len(test_set))
    last_col_idx = len(test_set.columns) - 1
    # prepare model
    summaries = summarize_by_class(training_set)
    # test model
    test_set_list = test_set.values.tolist()
    predictions = get_predictions(summaries, test_set_list)
    y_val = test_set.iloc[:,last_col_idx].values.tolist()
    accuracy = get_accuracy(y_val, predictions)
    print('\nAccuracy: {0}%').format(accuracy)
    #Confusion Matrix
    unique, matrix = confusion_matrix(y_val,predictions)
    #print_confusion_matrix(unique, matrix)
    #print('\n')
    plot_confusion_matrix(np.array(matrix),unique)


if __name__ == '__main__':
    print 'Naive Bayes'
    filename = sys.argv[1]
    main(filename)
