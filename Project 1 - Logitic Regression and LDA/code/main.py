'''
main script. The primary output of the supporting modules is calculated here
Specifically, this script Initializes the models defined in the other modules,
collects data from the predefined data sources, and then runs several
experiments on the models using the provided data.
'''

import helpers
from lda import lda
from log_regress import LogisticRegression
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt

if __name__ == '__main__':
    # Collect Datasets
    print("Collecting datasets from csv files")
    # Y index indicates which column represents the output
    cancerYIndex = 10
    cancerData = helpers.readData("data/breast-cancer-wisconsin.data", ',', cancerYIndex, 3.0)

    wineYIndex = 11
    wineData = helpers.readData("data/winequality-red.csv", ';', wineYIndex, 5.0)

    print("Formatting data")
    # Collect data and split into X and Y features and outputs
    data = [(wineData, wineYIndex), (cancerData, cancerYIndex)]
    datasets = []
    for item in data:
        # Split and reunite data around mid columns
        # Unnescessary for these datasets, but here anyway
        leftX = item[0][:, :item[1]]
        if (item[1] < np.shape(item[0])[1]):
            # If more features exist on the right side of Y
            rightX = item[0][:, item[1]+1:]
            X = np.concatenate((leftX, rightX), axis=1)
        else:
            # Y is end of matrix
            X = leftX
        Y = item[0][:, item[1]]
        datasets.append((X, Y))

    # Initialize models
    print("Initializing models")
    ldaModel = lda()
    logModel = LogisticRegression()

    # Perform tests on models
    print("Running tests on models")
    print("*----------------------------------------*")

    #-------------------------------------#

    print("\nTest 1: Different learning rates on models\n")
    # test different learning rates. These ones are manually inputted
    tests = [
        (logModel, datasets[0], "Logistic Regression test on wine dataset", 0.005),
        (logModel, datasets[0], "Logistic Regression test on wine dataset", 0.01),
        (logModel, datasets[0], "Logistic Regression test on wine dataset", 0.025),
        (logModel, datasets[0], "Logistic Regression test on wine dataset", 0.05),
        (logModel, datasets[0], "Logistic Regression test on wine dataset", 0.075),
        (logModel, datasets[0], "Logistic Regression test on wine dataset", 0.525),
        (logModel, datasets[0], "Logistic Regression test on wine dataset", 0.55),
        (logModel, datasets[0], "Logistic Regression test on wine dataset", 0.45),
        (logModel, datasets[0], "Logistic Regression test on wine dataset", 0.475)
    ]

    # Populate experiment 1 with more data for better graph, automated addition
    count = 0.25
    inc = 0.25;
    while(count < 20):
        tests.append((logModel, datasets[0], "Logistic Regression test on wine dataset", count))
        count += inc

    points = []
    for test in tests:
        #Run Test
        startTime = datetime.now(tz=None)
        res = helpers.kFold(test[0], test[1][0], test[1][1], 5, shuffle=False, params=[test[3], 1000, 1000])
        endTime = datetime.now(tz=None)

        points.append((test[3], res))

        print(test[2])
        print("Learning rate: " + str(test[3]))
        print("Test result: " + str(res))
        print("Time taken: " + str(endTime - startTime) + "\n")

    #-------------------------------------#

    print("\nTest 2: Compare running times of Logistic Regression vs Linear Discriminant Analysis\n")
    # Compare runtime and accuracy of data on both datasets
    tests = [
        (ldaModel, datasets[0], "Linear Discriminant Analysis test on wine dataset"),
        (ldaModel, datasets[1], "Linear Discriminant Analysis test on cancer dataset"),
        (logModel, datasets[0], "Logistic Regression test on wine dataset"),
        (logModel, datasets[1], "Logistic Regression test on cancer dataset")
    ]
    for test in tests:
        #Run Test
        startTime = datetime.now(tz=None)
        res = helpers.kFold(test[0], test[1][0], test[1][1], 5)
        endTime = datetime.now(tz=None)

        print(test[2])
        print("Test result: " + str(res))
        print("Time taken: " + str(endTime - startTime) + "\n")

    #-------------------------------------#

    print("\nTest 3: Finding new subset of features in wine dataset to improve accuracy\n")

    yindex = data[0][1] - 1
    k_fold_acc = [("Default params", helpers.kFold(logModel, datasets[0][0], datasets[0][1], 5, shuffle=False))]

    # Prepare optimal dataset
    p = [2, 400, 1000]
    X_new = datasets[0][0]
    # Remove a feature that doesn't seem to contribute to accuracy
    X_new = np.delete(X_new, 8, axis=1)

    # Calculate accuracy of new params
    k_fold_acc.append(("Optimal params", helpers.kFold(logModel, X_new, datasets[0][1], 5, shuffle=False, params=p)))

    [print (x) for x in k_fold_acc]

    # Print experiement 1 to screen
    if points != []:
        # Transpose points
        xy = list(map(list, zip(*points)))

        # Prepare chart
        plt.scatter(xy[0], xy[1], label="test", color="black")

        plt.xlabel('Learning Rate')
        plt.ylabel('Accuracy')

        plt.title("Figure 1: Learning Rate vs Accuracy")
        print("Press q to quit")
        plt.show()
    plt.close()

    print("Done")
