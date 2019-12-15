'''
Helper class to implement commonly used methods.
Including methods for reading data and k-fold cross examination
'''

import numpy as np
import matplotlib.pyplot as plt
import csv
from test import Test

def getRowItem(row, expectedSize, replaceIndex, thresh):
    '''
    Given a row from a csv reader and the expected size of the row, returns the
    row in a form that can be put into an np array, or none is data is not valid
    '''
    item = []
    if len(row) is not expectedSize:
        # Skip row, missing data in row
        return None
    for i in range(len(row)):
        try:
            fltField = float(row[i])
            if (i == replaceIndex):
                if (fltField > thresh):
                    item.append(1)
                else:
                    item.append(0)
            else:
                item.append(fltField)
        except ValueError:
            # Failed to convert value to float, invalid item
            return None
    return item

def cleanData(data):
    """
    Performs cleaning operations on the data as desired

    Empty, no suitable function was determined.
    """
    return data

def readData(file, delim, replaceIndex, thresh):
    """
    Reads the data and completes some simple validation to the data
    """
    with open(file) as csvfile:
        readCSV = csv.reader(csvfile, delimiter=delim)

        items = []

        for row in readCSV:
            item = getRowItem(row, len(row), replaceIndex, thresh)
            if (item is not None):
                # add to array if getItem returned a row
                items.append(item)

        E = np.array(items)

        # Normalize data
        for j in range(len(items[0])):
            E[:, j] = (E[:, j] - np.amin(E[:, j])) / (np.max(E[:, j]) - np.min(E[:, j]))

        cleanedArr = cleanData(E)
        return cleanedArr

def npShuffleTogether(a, b):
    '''
    Uses built-in np permutation generator to shuffle arrays together such that
    a and b still hold their original 1-1 correlation
    '''
    p = np.random.permutation(len(a))
    return a[p], b[p]

def kFold(model, x, y, k, shuffle=True, params=[]):
    '''
    K-fold cross validation. Splits the input and output sets X and Y into K
    folds. Each individual fold is used as a validation set on the model
    fitted by the remaining data.
    X and Y are assumed to have a 1-1 link of inputs to outputs
    (F: X[i] -> Y[i])
    '''
    # Shuffle Data
    if (shuffle):
        x, y = npShuffleTogether(x, y)

    # Split the data
    trainFolds = np.array_split(x, k)
    outFolds = np.array_split(y, k)

    res = []
    # Iterate through the segments
    for i in range(k):
        # merge everything that isn't current fold into its own array
        trainX = [trainFolds[j] for j in range(k) if j != i]
        outY = [outFolds[j] for j in range(k) if j != i]

        xMinusK = np.concatenate(trainX, axis=0)
        yMinusK = np.concatenate(outY, axis=0)

        # Fit this subset of the data
        if (params != []):
            model.fit(xMinusK, yMinusK, params)
        else:
            model.fit(xMinusK, yMinusK)

        # Test model with fold that wasn't used for training
        res.append(model.evaluate_acc(trainFolds[i], outFolds[i]))
    try:
        return sum(res) / len(res)
    except:
        # Model is returning None rather than an float which means data is bad
        return None

if __name__ == '__main__':
    readData("./data/winequality-red.csv", ';', 11, 5.0)

    testModel = Test()
    x = np.array([[1, 2], [3, 4], [5, 6]])
    y = np.array([0, 1, 2])

    res = kFold(testModel, x, y, 5)
    print(res)
