'''
Logistic Regression machine learning model implementation
'''

import numpy as np
import matplotlib.pyplot as plt
import csv
import math

import helpers
from model import Model


class LogisticRegression(Model):

    def __init__(self):
        self.fitted = False
        super().__init__()

    # Calculate sigmoid - used to map real values between 0 and 1
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    # Weighted sum of inputs
    def weighted_inputs(self, x, theta):
        return np.dot(x, theta)

    # The probability that results from the sigmoid function
    def predicted_prob(self, x, theta):
        return self.sigmoid(self.weighted_inputs(x, theta))

    # Computes the cost of the vectorized cross-entropy[a.k.a log loss] function
    def cost(self, x, y, theta):
        correction = 1e-5
        m = x.shape[0]
        tcost = -(1 / m) * np.sum(
            y * np.log(self.predicted_prob(x, theta) + correction) + (1 - y) *
                np.log(1 - self.predicted_prob(x, theta) + correction))
        return tcost

    # computes gradient of cost function at theta
    def gradient(self, x, y, theta):
        m = x.shape[0]
        return (1 / m) * np.dot(x.T, self.sigmoid(
            self.weighted_inputs(x, theta)) - y)

    def fit(self, x, y, params=[0.5, 1000, 1000]):
        self.lr = params[0]
        self.n_iter = params[1]
        start_loss = params[2]
        theta = np.zeros([x.shape[1]])
        self.return_theta = theta
        new_loss = self.cost(x, y, theta) + start_loss

        # gradient descent for logistic regression
        for i in range(self.n_iter):
            grad1 = self.gradient(x, y, theta)
            theta = theta - (self.lr * grad1)
            loss = self.cost(x, y, theta)
            if (loss < new_loss):
                new_loss = loss
                self.return_theta = theta
            else:
                continue

        self.fitted = True
        return

    def predict(self, x, y):
        theta = self.return_theta
        theta = theta[:, np.newaxis]
        prob = self.predicted_prob(x, theta)

        # If probability is above 0.5, then prediction is 1. Else 0
        temp = 0
        if prob >= 0.5:
            temp = 1
        return temp == int(y)

    def evaluate_acc(self, X, Y):
        if (not self.fitted):
            return None

        # Predicts and stores result for each item
        res = []
        size = np.shape(X)[0]
        for i in range(size):
            # Predict each result, store in list
            res.append(self.predict(X[i], Y[i]))

        # Calulate percentage from list
        total = len (res)
        countAccurate = sum(res)

        return (countAccurate / total) * 100

if __name__ == '__main__':
    # yindex = 10
    # data = helpers.readData("data/breast-cancer-wisconsin.data", ',', yindex, 3.0)
    yindex = 11
    data = helpers.readData("data/winequality-red.csv", ';', yindex, 5.0)

    # Slice last column to be output, except for last item which will be used
    # in prediction. If statement merges dataset back together ignoring Y
    leftX = data[:, :yindex]
    if yindex < np.shape(data)[1]:
        # If more data points exist on the right side of Y
        rightX = data[:, yindex+1:]
        X = np.concatenate((leftX, rightX), axis=1)
    else:
        # Y is final element in matrix
        X = leftX
    Y = data[:, yindex]

    # Create LogisticRegression instance
    logRegressModel = LogisticRegression()

    # Use last 10 elements as validation set
    print("Fitting data")
    logRegressModel.fit(X[:-10, :], Y[:-10])

    # Predict on last element in set which wasn't used as training data
    print('Fitted, making simple prediction')
    print(logRegressModel.predict(X[-1, :], Y[-1]))

    print('Evaluating Accuracy by omitting parameter number')
    print(logRegressModel.evaluate_acc(X[-10:, :], Y[-10:]))

    print('k-fold cross examination')
    print(helpers.kFold(logRegressModel, X, Y, 5))
