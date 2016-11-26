import math
import csv


'''
Load a CSV file
'''
def load_csv(filename):
    lines = csv.reader(open(filename, "rb"))
    dataset = list(lines)
    return dataset


'''
Convert string column to integer
'''
def str_column_to_int(dataset, column):
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
Rescale dataset columns to the range 0-1
'''
def normalize_dataset(dataset, minmax):
    for row in dataset:
        for i in range(len(row)):
            row[i] = (row[i] - minmax[i][0]) / (minmax[i][1] - minmax[i][0])

'''
calculate the Euclidean distance between two vectors
'''
def euclidean_distance(row1, row2):
    distance = 0.0
    for i in range(len(row1)-1):
        distance += (row1[i] - row2[i]) ** 2
    return math.sqrt(distance)


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
