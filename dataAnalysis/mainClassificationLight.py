import sys

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from mne.decoding import CSP
from estimationModels.dataLoader import loadData




def computeCSP(x_train, x_test, y_train):
    x_train = np.array(x_train).transpose((0, 2, 1))
    x_test = np.array(x_test).transpose((0, 2, 1))

    csp = CSP(n_components=8, reg="oas", log=True)
    csp.fit(np.array(x_train), np.array(y_train))

    x_fold_train = csp.transform(np.array(x_train))
    x_fold_test = csp.transform(np.array(x_test))
    return x_fold_train, x_fold_test





def classificationTests(X, Y, processData=lambda xtrain, xtest, ytrain: (xtrain, xtest)):
    print("----")
    print("Intra subject estimation")
    meanScores = np.zeros(5)
    for i in range(len(X)):
        print("    estimation for", i)
        x = np.concatenate(X[i])
        y = np.concatenate(Y[i])

        crossValScores = np.zeros(5)
        for k in range(5):
            b1, b2 = int(len(x) * 0.2 * k), int(len(x) * 0.2 * (k + 1))
            x_fold_train = np.concatenate((x[:b1], x[b2:]))
            x_fold_test = np.array(x[b1:b2])
            y_fold_train = np.concatenate((y[:b1], y[b2:]))
            y_fold_test = np.array(y[b1:b2])

            x_fold_train, x_fold_test = processData(x_fold_train, x_fold_test, y_fold_train)

            models = [RandomForestClassifier(n_estimators=10),
                      KNeighborsClassifier(n_neighbors=10),
                      LogisticRegression(max_iter=1000),
                      LinearDiscriminantAnalysis(),
                      MLPClassifier(hidden_layer_sizes=(100, 50), learning_rate="invscaling", learning_rate_init=0.001, max_iter=200, verbose=False, shuffle=True)]
            scores= [0]*len(models)
            for m in range(len(models)):
                model = models[m]
                model.fit(x_fold_train, y_fold_train)
                scores[m] = model.score(x_fold_test, y_fold_test)

            crossValScores += np.array(([np.mean(s) for s in scores]))

        print("scores subject : ", crossValScores/5)

        meanScores += crossValScores/5
    print("    Mean Scores")
    print(meanScores / len(X))

    print("----")
    print("Inter subject estimation 1 (isolate session)")
    x = []
    y = []
    for i in range(10):
        for j in range(len(X)):
            if i < len(X[j]):
                x.extend(X[j][i])
                y.extend(Y[j][i])

    x = np.array(x)

    crossValScores = np.zeros(5)
    for k in range(5):
        b1, b2 = int(len(x) * 0.2 * k), int(len(x) * 0.2 * (k + 1))
        x_fold_train = np.concatenate((x[:b1], x[b2:]))
        x_fold_test = np.array(x[b1:b2])
        y_fold_train = np.concatenate((y[:b1], y[b2:]))
        y_fold_test = np.array(y[b1:b2])

        x_fold_train, x_fold_test = processData(x_fold_train, x_fold_test, y_fold_train)

        models = [RandomForestClassifier(n_estimators=10),
                  KNeighborsClassifier(n_neighbors=10),
                  LogisticRegression(max_iter=1000),
                  LinearDiscriminantAnalysis(),
                  MLPClassifier(hidden_layer_sizes=(100, 50), learning_rate="invscaling", learning_rate_init=0.001,
                                max_iter=200, verbose=False, shuffle=True)]
        scores = [0] * len(models)
        for m in range(len(models)):
            model = models[m]
            model.fit(x_fold_train, y_fold_train)
            scores[m] = model.score(x_fold_test, y_fold_test)

        crossValScores += np.array(([np.mean(s) for s in scores]))
    print("scores Inter subject 1 : ", crossValScores / 5)

    print("----")
    print("Inter subject estimation 2 (isolate subject)")
    x = []
    y = []
    for i in range(len(X)):
        x.extend(np.concatenate(X[i]))
        y.extend(np.concatenate(Y[i]))

    x = np.array(x)

    crossValScores = np.zeros(5)
    for k in range(5):
        b1, b2 = int(len(x) * 0.2 * k), int(len(x) * 0.2 * (k + 1))
        x_fold_train = np.concatenate((x[:b1], x[b2:]))
        x_fold_test = np.array(x[b1:b2])
        y_fold_train = np.concatenate((y[:b1], y[b2:]))
        y_fold_test = np.array(y[b1:b2])

        x_fold_train, x_fold_test = processData(x_fold_train, x_fold_test, y_fold_train)

        models = [RandomForestClassifier(n_estimators=10),
                  KNeighborsClassifier(n_neighbors=10),
                  LogisticRegression(max_iter=1000),
                  LinearDiscriminantAnalysis(),
                  MLPClassifier(hidden_layer_sizes=(100, 50), learning_rate="invscaling", learning_rate_init=0.001,
                                max_iter=200, verbose=False, shuffle=True)]
        scores = [0] * len(models)
        for m in range(len(models)):
            model = models[m]
            model.fit(x_fold_train, y_fold_train)
            scores[m] = model.score(x_fold_test, y_fold_test)

        crossValScores += np.array(([np.mean(s) for s in scores]))
    print("scores Inter subject 2 : ", crossValScores / 5)






if __name__ == "__main__":
    arg = sys.argv
    DATA_FOLDER_PATH = ""
    if len(arg) < 2:
        quit()
    else:
        DATA_FOLDER_PATH = arg[1]

    print("TESTS using TDF")
    X, Y = loadData(DATA_FOLDER_PATH, (0, 1), datasetType=("sign", 5), dataType="TDF")
    classificationTests(X, Y)

    print("TESTS using CSP")
    X, Y = loadData(DATA_FOLDER_PATH, (0, 1), datasetType=("sign", 5), dataType="CSP")
    classificationTests(X, Y, processData=computeCSP)

