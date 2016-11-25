import csv
import sys
import random
import math
import pandas as pd

def load_csv(filename):
    lines = csv.reader(open(filename, "rb"))
    '''
    lines type :  <type '_csv.reader'>
    '''
    print '\nlines type : ', type(lines)
    '''
    use pandas to read csv file
    l type :  <class 'pandas.core.frame.DataFrame'>
    '''
    l = pd.read_csv(filename)
    print '\nl type : ', type(l)

    dataset = list(lines)
    for i in range(len(dataset)):
        dataset[i] = [float(x) for x in dataset[i]]
    return dataset

def split_dataset(dataset, split_ratio):
    train_size = int(len(dataset) * split_ratio)
    train_set = []
    copy = list(dataset)
    while len(train_set) < train_size:
        index = random.randrange(len(copy))
        train_set.append(copy.pop(index))

    return train_set, copy

def separate_by_class(dataset):
    separated = {}
    for i in range(len(dataset)):
        vector = dataset[i]
        if (vector[-1] not in separated):
            separated[vector[-1]] = []
        separated[vector[-1]].append(vector)
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

    #What is this for?
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
    exponent = math.exp(-(math.pow(x-mean,2)/(2*math.pow(stdev,2))))
    return (1 / (math.sqrt(2*math.pi) * stdev)) * exponent



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
    for x in range(len(test_set)):
        if test_set[x][-1] == predictions[x]:
            correct += 1
    return (correct/float(len(test_set))) * 100.0

# calculate a confusion matrix
def confusion_matrix(actual, predicted):
    #print 'actual : ', actual
    #print 'predicted : ', predicted
    actuals = []
    for x in range(len(actual)):
        actuals.append(actual[x][-1])
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
    print('(P)' + ' '.join(str(x) for x in unique))
    print('(A)---')
    for i, x in enumerate(unique):
        print("%s| %s" % (x, ' '.join(str(x) for x in matrix[i])))

def main(filename):
    split_ratio = 0.70
    dataset = load_csv(filename)
    training_set, test_set = split_dataset(dataset, split_ratio)
    print('Split {0} rows into train={1} and test={2} rows').format(len(dataset), len(training_set), len(test_set))
    # prepare model
    summaries = summarize_by_class(training_set)
    # test model
    predictions = get_predictions(summaries, test_set)
    accuracy = get_accuracy(test_set, predictions)
    print('Accuracy: {0}%').format(accuracy)

    #Confusion Matrix
    unique, matrix = confusion_matrix(test_set,predictions)
    print_confusion_matrix(unique, matrix)

if __name__ == '__main__':
    print 'Naive Bayes'
    '''
    test_dataset = [[1, 20, 1], [2, 21, 0], [3, 22, 1]]
    separated = separate_by_class(test_dataset)
    print('Separated instances: {0}').format(separated)

    #Test mean and stdev
    test_numbers = [1, 2, 3, 4, 5]
    print('Summary of {0}: mean={1}, stdev={2}').format(test_numbers, mean(test_numbers), stdev(test_numbers))

    # Test summarize
    test_dataset = [[1, 20, 0], [2, 21, 1], [3, 22, 0]]
    summary = summarize(test_dataset)
    print('Attribute summaries: {0}').format(summary)

    #Test summarize_by_class
    test_dataset = [[1, 20, 1], [2, 21, 0], [3, 22, 1], [4, 22, 0], [5, 23, 0]]
    summary = summarize_by_class(test_dataset)
    #summary = summarize(test_dataset)
    print('Summary by class value: {0}').format(summary)


    #Test calculate_probability
    x = 71.5
    mean = 73
    stdev = 6.2
    probability = calculate_probability(x, mean, stdev)
    print('Probability of belonging to this class: {0}').format(probability)

    #Test calculate_class_probabilities
    summaries = {0: [(1, 0.5)], 1: [(20, 5.0)]}
    input_vector = [1.1, '?']
    probabilities = calculate_class_probabilities(summaries, input_vector)
    print('Probabilities for each class: {0}').format(probabilities)

    #Test predict()
    summaries = {'A': [(1, 0.5)], 'B': [(20, 5.0)]}
    #input_vector = [1.1, '?']
    input_vector = [1.1, '?']
    result = predict(summaries, input_vector)
    print('Prediction: {0}').format(result)

    #Test get_prediction()
    summaries = {'A': [(1, 0.5)], 'B': [(20, 5.0)]}
    test_set = [[1.1, '?'], [19.1, '?']]
    predictions = get_predictions(summaries, test_set)
    print('Predictions: {0}').format(predictions)

    #Test get_accuracy()
    test_set = [[1, 1, 1, 'a'], [2, 2, 2, 'a'], [3, 3, 3, 'b']]
    predictions = ['a', 'a', 'a']
    accuracy = get_accuracy(test_set, predictions)
    print('Accuracy: {0}').format(accuracy)
    '''


    '''
    # Test confusion matrix with integers
    actual = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1,3,5]
    predicted = [0, 1, 1, 0, 0, 1, 0, 1, 1, 1,0,5]
    unique, matrix = confusion_matrix(actual, predicted)
    print(unique)
    print(matrix)
    print_confusion_matrix(unique, matrix)
    '''

    '''
    Run the algo for actual dataset
    '''
    filename = sys.argv[1]
    main(filename)




