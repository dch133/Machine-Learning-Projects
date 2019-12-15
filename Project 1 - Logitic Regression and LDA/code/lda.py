'''
Implementation of Linear Discriminant Analysis (LDA)
'''

import numpy as np
import matplotlib.pyplot as plt
import csv
import math

import helpers
from model import Model


class lda(Model):

    def __init__(self):
        self.fitted = False
        super().__init__()

    # Estimate P(y), µ (mean), Σ (covariance) from the training data, then apply log-odds ratio
    def fit(self, X, Y, params=[]):
        # 1) Calculate Mean for every feature for every entry
        # Create an entries matrix without the class column
        F = X

        # Create index on Classes
        ind = Y
        ind = ind.astype(int)

        mu_1 = F[ind == 1].mean(axis=0)
        mu_0 = F[ind == 0].mean(axis=0)

        entries_1 = F[ind == 1]
        entries_0 = F[ind == 0]

        N1 = entries_1.shape[0]
        N0 = entries_0.shape[0]

        # 2) Estimate P(y): Probability of a class
        P_y_1 = N1 / (N0 + N1)
        P_y_0 = N0 / (N0 + N1)

        # 3) Calculate Covariance for all features (num_features X num_features Matrix)
        x_i_minus_mu_1 = []
        x_i_minus_mu_0 = []
        x_i_minus_mu_1 = np.subtract(entries_1, np.array(mu_1))
        x_i_minus_mu_0 = np.subtract(entries_0, np.array(mu_0))
        x_i_minus_mu_1_trans = np.transpose(x_i_minus_mu_1)
        x_i_minus_mu_0_trans = np.transpose(x_i_minus_mu_0)

        sigma_1 = (np.dot(x_i_minus_mu_1_trans, x_i_minus_mu_1) / (N0 + N1 - 2))
        sigma_0 = (np.dot(x_i_minus_mu_0_trans, x_i_minus_mu_0) / (N0 + N1 - 2))

        # Store fitted values in instance
        self.P_y_0 = P_y_0
        self.P_y_1 = P_y_1
        self.mu_0 = mu_0
        self.mu_1 = mu_1
        self.sigma = sigma_0 + sigma_1
        self.F = F

        self.fitted = True

        return

    def predict(self, x, y):
        if (not self.fitted):
            return None

        # Retrieve fitted variables
        P_y_1 = self.P_y_1
        P_y_0 = self.P_y_0
        mu_1 = self.mu_1
        mu_0 = self.mu_0
        sigma = self.sigma

        # Calculate ouput
        entries_selected = x

        sigma_inv = np.linalg.inv(sigma)
        mu_trans_1 = np.transpose(mu_1)
        mu_trans_0 = np.transpose(mu_0)

        log_P = math.log(P_y_1 / P_y_0)
        w_0 = log_P - 0.5 * np.array(mu_1.dot(sigma_inv)).dot(mu_trans_1) + 0.5 * np.array(mu_0.dot(sigma_inv)).dot(
            mu_trans_0)
        x_trans_w = np.array(entries_selected.dot(sigma_inv)).dot(np.subtract(mu_trans_1, mu_trans_0))
        prediction = w_0 + x_trans_w

        # If prediction is positive, then expected output is 1
        y_out = 0
        if(prediction > 0):
            y_out = 1

        if(y_out == int(y)):
            return 1
        else:
            return 0

    def evaluate_acc(self, X, Y):
        if (not self.fitted):
            return None

        # Calculate percentage total
        res = []
        size = np.shape(X)[0]
        for i in range(size):
            # Predict each result, store in list
            res.append(self.predict(X[i], Y[i]))

        # Calulate percentage from list
        total = len (res)
        # countAccurate = len ([1 for x in res if x==1])
        countAccurate = sum(res)

        return (countAccurate / total) * 100


if __name__ == '__main__':
    # read the data
    # yindex = 10
    # data = helpers.readData("./data/breast-cancer-wisconsin.data", ',', yindex, 3.0)
    yindex = 11
    data = helpers.readData("data/winequality-red.csv", ';', yindex, 5.0)


    # Slice last column to be output, except for last item which will be used
    # in prediction
    leftX = data[:, :yindex]
    if (yindex < np.shape(data)[1]):
        # If more data points exist on the right side of Y
        rightX = data[:, yindex+1:]
        X = np.concatenate((leftX, rightX), axis=1)
    else:
        # Y is end of matrix
        X = leftX
    # X = data[:-10, :-1]
    Y = data[:, yindex]

    # Create lda instance
    ldaModel = lda()

    # Use last 10 elements as validation set
    print("Fitting data")
    ldaModel.fit(X[:-10, :], Y[:-10])

    # Predict on last element in set which wasn't used as training data
    print('Fitted, making simple prediction')
    print(ldaModel.predict(X[-1, :], Y[-1]))

    print('Evaluating Accuracy by omitting parameter number')
    print(ldaModel.evaluate_acc(X[-10:, :], Y[-10:]))

    print('k-fold cross examination')
    print(helpers.kFold(ldaModel, X, Y, 5))
