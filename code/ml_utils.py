import math
import csv
import random


'''
Load a CSV file
'''
def load_csv(filename):
    lines = csv.reader(open(filename, "rb"))
    dataset = list(lines)
    return dataset


'''
Convert string column to integer
dataset and column position are inputs
To convert multiple columns, it has to be called multiple times
'''
def cat_str_column_to_int(dataset, column):
    class_values = [row[column] for row in dataset]
    unique = set(class_values)
    lookup = dict()
    for i, value in enumerate(unique):
        lookup[value] = i
    #print 'lookup:\n', lookup
    for row in dataset:
        row[column] = lookup[row[column]]
    #print 'dataset after looked up : \n', dataset
    return lookup, dataset


'''
Convert all the columns to float
Use str_column_to_int(dataset, column) method to convert any categorical variable to numerical before applying this method
'''
def str_columns_to_float(dataset, col_start=0, col_end=999):
    #Following code will produce a dataset where the first col will start from specified col_start
    #for i in range(len(dataset)):
    #    dataset[i] = [float(dataset[i][j]) for j in range(col_start, len(dataset[i]))]

    for i in range(len(dataset)):
        dataset[i] = [float(x) for x in dataset[i]]
    return dataset


'''
Find the min and max values for each column
'''
def dataset_minmax(dataset):
    minmax = list()
    for i in range(len(dataset[0])):
        col_values = [row[i] for row in dataset]
        value_min = min(col_values)
        value_max = max(col_values)
        minmax.append([value_min, value_max])
    return minmax


'''
Rescale dataset columns to the range 0-1
Avoid devide by zero situation
'''
def normalize_dataset(dataset, minmax):
    for row in dataset:
        for i in range(len(row)):
            row[i] = (row[i] - minmax[i][0]) / (minmax[i][1] - minmax[i][0])
    return dataset

'''
Split Dataset into two sets - training and test - based on ratio given
'''
def split_dataset(dataset, split_ratio):
    train_size = int(len(dataset) * split_ratio)
    train_set = []
    test_set = list(dataset)
    while len(train_set) < train_size:
        index = random.randrange(len(test_set))
        train_set.append(test_set.pop(index))

    return train_set, test_set

'''
Split a dataset into k folds
'''
def cross_validation_split(dataset, n_folds):
    dataset_split = list()
    dataset_copy = list(dataset)
    fold_size = len(dataset) / n_folds
    for i in range(n_folds):
        fold = list()
        while len(fold) < fold_size:
            index = random.randrange(len(dataset_copy))
            fold.append(dataset_copy.pop(index))
        dataset_split.append(fold)
    return dataset_split


'''
Group the records based on available classes in Y
It will return a dict - key will be class values and value for this key will the records
'''
def group_by_class(dataset):
    separated = {}
    for i in range(len(dataset)):
        vector = dataset[i]
        if (vector[-1] not in separated):
            separated[vector[-1]] = []
        separated[vector[-1]].append(vector)
    return separated

'''
Calculate mean of numbers in a vector
'''
def mean(numbers):
    return sum(numbers)/float(len(numbers))

'''
Calculate Standard Deviation of numbers in a vector
'''
def stdev(numbers):
    avg = mean(numbers)
    variance = sum([pow(x - avg,2) for x in numbers])/float(len(numbers) - 1)
    return math.sqrt(variance)

'''
calculate the Euclidean distance between two vectors
'''
def euclidean_distance(row1, row2):
    distance = 0.0
    for i in range(len(row1)-1):
        distance += (row1[i] - row2[i]) ** 2
    return math.sqrt(distance)


'''
Calculate accuracy percentage
Input expected to be two vector - actual class and predicted class
'''
def accuracy_metric(actual, predicted):
    correct = 0
    for i in range(len(actual)):
        if actual[i] == predicted[i]:
            correct += 1
    return correct / float(len(actual)) * 100.0


'''
Calculate a confusion matrix
It will work for n dimention
'''
def confusion_matrix(actual, predicted):
    #print 'actual : ', actual
    #print 'predicted : ', predicted
    unique = set(actual)
    matrix = [list() for x in range(len(unique))]
    for i in range(len(unique)):
        matrix[i] = [0 for x in range(len(unique))]
    lookup = dict()
    for i, value in enumerate(unique):
        lookup[value] = i
        #lookup[i] = value
    #print 'lookup : ', lookup

    for i in range(len(actual)):
        x = lookup[actual[i]]
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