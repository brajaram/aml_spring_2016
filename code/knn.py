import ml_utils as utils
import random
import sys


'''
Evaluate an algorithm using a cross validation split
For each fold, calculate the Accuracy and generate the Confucion Matrix
'''
def evaluate_algorithm(dataset, algorithm, n_folds, *args):
    folds = utils.cross_validation_split(dataset, n_folds)
    scores = list()
    fold_no = 0
    for fold in folds:
        train_set = list(folds)
        train_set.remove(fold)
        train_set = sum(train_set, [])
        test_set = list()
        for row in fold:
            row_copy = list(row)
            test_set.append(row_copy)
            row_copy[-1] = None
        predicted = algorithm(train_set, test_set, *args)
        actual = [row[-1] for row in fold]
        accuracy = utils.accuracy_metric(actual, predicted)
        scores.append(accuracy)
        print '\nAccuracy of fold #{0} is  : {1}'.format(fold_no, accuracy)
        #print '\nActual : \n', actual
        #print '\nPredicted : \n', predicted
        unique, matrix = utils.confusion_matrix(actual, predicted)
        print '\nConfusion Matrix for fold #{0} is:'.format(fold_no)
        utils.print_confusion_matrix(unique, matrix)
        fold_no += 1

    return scores



'''
Locate most similar neighbours based in Euclidean distance
Number of Neighbours to be selected is parameterized
'''
def get_neighbors(train, test_row, num_neighbors):
    distances = list()
    for train_row in train:
        dist = utils.euclidean_distance(test_row, train_row)
        #A list of tuples
        distances.append((train_row, dist))
    distances.sort(key=lambda tup: tup[1])
    neighbors = list()
    for i in range(num_neighbors):
        neighbors.append(distances[i][0])
    return neighbors

'''
KNN for classification
Make a classification prediction with neighbors
'''
def predict_classification(train, test_row, num_neighbors):
    neighbors = get_neighbors(train, test_row, num_neighbors)
    output_values = [row[-1] for row in neighbors]
    prediction = max(set(output_values), key=output_values.count)
    return prediction


'''
KNN for Regression
Make a prediction with neighbors
'''
def predict_regression(train, test_row, num_neighbors):
    neighbors = get_neighbors(train, test_row, num_neighbors)
    output_values = [row[-1] for row in neighbors]
    prediction = sum(output_values) / float(len(output_values))
    return prediction


'''
The Algorithm
It takes the training set and test set and number of neighbours as input arguments.
It generates prediction for each input record in test set
'''
def k_nearest_neighbors(train, test, num_neighbors):
    predictions = list()
    for row in test:
        output = predict_classification(train, row, num_neighbors)
        predictions.append(output)
    return predictions

def main(filename, cat_string_cols_pos=8, n_folds=5, num_neighbors=5):
    dataset = utils.load_csv(filename)
    cols = cat_string_cols_pos.split(',')
    if cols[0].lower() == 'na':
        print '\nFile does not have a categorical string variable'
    else:
        for col in cols:
            lookup, dataset = utils.cat_str_column_to_int(dataset, int(col))
            print 'Unique Values in {0}th col is : {1}'.format(col, lookup)
    #lookup, dataset = utils.cat_str_column_to_int(dataset, 8)
    #print 'Unique Values in 0th col : ', lookup
    dataset = utils.str_columns_to_float(dataset)
    #print 'dataset with float values:\n', dataset

    # evaluate algorithm
    scores = evaluate_algorithm(dataset, k_nearest_neighbors, n_folds, num_neighbors)
    print('Scores: %s' % scores)
    print('Mean Accuracy: %.3f%%' % (sum(scores) / float(len(scores))))

if __name__ == '__main__':
    print 'KNN Begins\n'
    print 'Last Column of the Dataset has to be the predicted columns\n'
    print '\n******Arguments******\n1. Filename (mandatory)\n2. Column Positions for Categorical String variables as comma separated values(optional)'
    print '3. N for N fold validation (optional)\n4. K for selecting K neoghbours (optional)\n'
    print '\n****#2 is mandatory if subsequent parameter is to be passed. In that case, pass it as na if there is not string variable*****'
    #ToDo : Improve the above feature

    random.seed(1)
    filename = sys.argv[1]
    print '\nFile Name is : ', filename
    #filename = 'abalone.csv'
    cat_string_cols_pos = sys.argv[2]
    print '\ncat_string_cols_pos : ', cat_string_cols_pos
    main(filename, cat_string_cols_pos)

