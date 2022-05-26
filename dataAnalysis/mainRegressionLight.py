import sys
import numpy as np

from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor

from estimationModels.dataLoader import loadData
from sklearn.metrics import mean_absolute_error, mean_squared_error
from mne.decoding import CSP
import random
from copy import copy

def estimationScore(actual, p, scoreRandom=0):
    actual = np.delete(actual, 13, 0)
    p = np.delete(p, 13, 0)

    ranges = [max(actual[i]) - min(actual[i]) for i in range(len(actual))]
    mae = mean_absolute_error(actual.transpose(), p.transpose(), multioutput='raw_values')

    score = 100 - 100 * np.mean(mae / ranges)
    score = round((score - scoreRandom) * 100 / (100-scoreRandom), 2)
    mae = round(np.mean(mae), 2)
    MSE = mean_squared_error(actual.transpose(), p.transpose())

    correlation = np.nanmean([np.corrcoef(actual.transpose()[i], p.transpose()[i])[0, 1] for i in range(len(p[0]))])
    NMSE = MSE / np.std(actual.transpose())

    return MSE, mae, score, correlation, NMSE


def estimateScoreFromRandom(y):
    p = []
    for channel in y:
        c = copy(channel)
        random.shuffle(c)
        p.append(c)
    MSE, mae, score, correlation, NMSE = estimationScore(y, np.array(p))
    return score


CSP_FIT_DATA_STEP = 10
def computeCSP(x_train, x_test, y_train):
    x_train = np.array(x_train).transpose((0, 2, 1))
    x_test = np.array(x_test).transpose((0, 2, 1))

    csp_train, csp_test = [], []
    for i in range(y_train.shape[1]):
        if i != 13:
            csp = CSP(n_components=8, reg="oas", log=True)
            csp.fit(np.array(x_train)[::CSP_FIT_DATA_STEP], np.array(y_train).transpose()[i].transpose()[::CSP_FIT_DATA_STEP] // 10)
            csp_train.append(csp.transform(np.array(x_train)))
            csp_test.append(csp.transform(np.array(x_test)))
    x_fold_train = np.concatenate(np.array(csp_train), axis=1)
    x_fold_test = np.concatenate(np.array(csp_test), axis=1)
    return x_fold_train, x_fold_test



def regressionTests(X, Y, processData=lambda xtrain, xtest, ytrain: (xtrain, xtest), testName="TDF"):
    print("----")
    print("Intra subject estimation")
    scores = []
    for i in range(len(X)):
        print("    estimation for", i)
        x = np.concatenate(X[i])
        y = np.concatenate(Y[i])

        scoreSubject = []
        for k in range(5):
            print("k = ", k)
            b1, b2 = int(len(x) * 0.2 * k), int(len(x) * 0.2 * (k + 1))
            x_fold_train = np.concatenate((x[:b1], x[b2:]))
            x_fold_test = np.array(x[b1:b2])
            y_fold_train = np.concatenate((y[:b1], y[b2:]))
            y_fold_test = np.array(y[b1:b2])

            x_fold_train, x_fold_test = processData(x_fold_train, x_fold_test, y_fold_train)

            scoreRandom = estimateScoreFromRandom(y_fold_train)

            print("LR")
            model = LinearRegression()
            model.fit(x_fold_train, y_fold_train)
            scoreLR = estimationScore(y_fold_test, model.predict(x_fold_test), scoreRandom)

            print("KNN")
            model = KNeighborsRegressor()
            model.fit(x_fold_train, y_fold_train)
            scoreKNN = estimationScore(y_fold_test, model.predict(x_fold_test), scoreRandom)

            print("RF")
            model = RandomForestRegressor(n_estimators=10)
            model.fit(x_fold_train, y_fold_train)
            scoreRF = estimationScore(y_fold_test, model.predict(x_fold_test), scoreRandom)

            print("MLP")
            model = MLPRegressor(hidden_layer_sizes=(100, 50), learning_rate="invscaling", learning_rate_init=0.001,
                                 max_iter=200, epsilon=0.5, early_stopping=True, verbose=False, shuffle=False)
            model.fit(x_fold_train, y_fold_train)
            scoreMLP = estimationScore(y_fold_test, model.predict(x_fold_test), scoreRandom)

            scoreSubject.append([scoreLR, scoreKNN, scoreRF, scoreMLP])
        scores.append(scoreSubject)
    print(scores)

    print("----")
    print("Inter subject estimation 1 (isolate session)")
    x = []
    y = []
    print(np.array(X).shape)
    for i in range(12):
        for j in range(len(X)):
            if i < len(X[j]):
                x.extend(X[j][i])
                y.extend(Y[j][i])

    x = np.array(x)
    y = np.array(y)
    scores = []
    for k in range(5):
        print("k = ", k)
        b1, b2 = int(len(x) * 0.2 * k), int(len(x) * 0.2 * (k + 1))
        x_fold_train = np.concatenate((x[:b1], x[b2:]))
        x_fold_test = np.array(x[b1:b2])
        y_fold_train = np.concatenate((y[:b1], y[b2:]))
        y_fold_test = np.array(y[b1:b2])

        x_fold_train, x_fold_test = processData(x_fold_train, x_fold_test, y_fold_train)

        scoreRandom = estimateScoreFromRandom(y_fold_train)

        print("LR")
        model = LinearRegression()
        model.fit(x_fold_train, y_fold_train)
        scoreLR = estimationScore(y_fold_test, model.predict(x_fold_test), scoreRandom)

        print("KNN")
        model = KNeighborsRegressor()
        model.fit(x_fold_train, y_fold_train)
        scoreKNN = estimationScore(y_fold_test, model.predict(x_fold_test), scoreRandom)

        print("RF")
        model = RandomForestRegressor(n_estimators=10)
        model.fit(x_fold_train, y_fold_train)
        scoreRF = estimationScore(y_fold_test, model.predict(x_fold_test), scoreRandom)

        print("MLP")
        model = MLPRegressor(hidden_layer_sizes=(100, 50), learning_rate="invscaling", learning_rate_init=0.001,
                             max_iter=200, epsilon=0.5, early_stopping=True, verbose=False, shuffle=False)
        model.fit(x_fold_train, y_fold_train)
        scoreMLP = estimationScore(y_fold_test, model.predict(x_fold_test), scoreRandom)

        scores.append([scoreLR, scoreKNN, scoreRF, scoreMLP])
    print(scores)

    print("----")
    print("Inter subject estimation 2 (isolate subject)")
    x = []
    y = []
    for i in range(len(X)):
        x.extend(np.concatenate(X[i]))
        y.extend(np.concatenate(Y[i]))
    x = np.array(x)
    y = np.array(y)
    scores = []
    for k in range(5):
        print("k = ", k)
        b1, b2 = int(len(x) * 0.2 * k), int(len(x) * 0.2 * (k + 1))
        x_fold_train = np.concatenate((x[:b1], x[b2:]))
        x_fold_test = np.array(x[b1:b2])
        y_fold_train = np.concatenate((y[:b1], y[b2:]))
        y_fold_test = np.array(y[b1:b2])

        x_fold_train, x_fold_test = processData(x_fold_train, x_fold_test, y_fold_train)

        scoreRandom = estimateScoreFromRandom(y_fold_train)

        print("LR")
        model = LinearRegression()
        model.fit(x_fold_train, y_fold_train)
        scoreLR = estimationScore(y_fold_test, model.predict(x_fold_test), scoreRandom)

        print("KNN")
        model = KNeighborsRegressor()
        model.fit(x_fold_train, y_fold_train)
        scoreKNN = estimationScore(y_fold_test, model.predict(x_fold_test), scoreRandom)

        print("RF")
        model = RandomForestRegressor(n_estimators=10)
        model.fit(x_fold_train, y_fold_train)
        scoreRF = estimationScore(y_fold_test, model.predict(x_fold_test), scoreRandom)

        print("MLP")
        model = MLPRegressor(hidden_layer_sizes=(100, 50), learning_rate="invscaling", learning_rate_init=0.001,
                           max_iter=200, epsilon=0.5, early_stopping=True, verbose=False, shuffle=False)
        model.fit(x_fold_train, y_fold_train)
        scoreMLP = estimationScore(y_fold_test, model.predict(x_fold_test), scoreRandom)

        scores.append([scoreLR, scoreKNN, scoreRF, scoreMLP])
    print(scores)


if __name__ == "__main__":
    arg = sys.argv
    DATA_FOLDER_PATH = ""
    if len(arg) == 2:
        DATA_FOLDER_PATH = arg[1]

    print("TEST  using tdf")
    X, Y = loadData(DATA_FOLDER_PATH, (0, 1), ("freemove", 6), 1, "TDF")
    regressionTests(X, Y, testName="TDF")

    print("TEST  using csp")
    X, Y = loadData(DATA_FOLDER_PATH, (0, 1), ("freemove", 6), 1, "CSP")
    regressionTests(X, Y, testName="CSP", processData=computeCSP)
