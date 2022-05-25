import sys
import numpy as np
import ezc3d
from os import path
import os
import matplotlib.pyplot as plt

from preprocessing.c3dRecovery import recoverFile
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.decomposition import PCA

from estimationModels.timeDomainFeatures import MAV, WL, RMS, MaximumAbsoluteAmplitude
from estimationModels.dataGenerators import createDatasetLabels
from estimationModels.featureSelection import MrmrRegression
from estimationModels.dataLoader import loadData
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from mne.decoding import CSP
from mne import set_log_level
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
    NMSE = MSE / np.var(actual.transpose())

    return MSE, mae, score, correlation, NMSE


def estimateScoreFromRandom(y):
    p = []
    for channel in y:
        c = copy(channel)
        random.shuffle(c)
        p.append(c)
    MSE, mae, score, correlation, NMSE = estimationScore(y, np.array(p))
    return score





import matplotlib.pyplot as plt
def featureAnalysisMrmr(x, y):


    featuresNames = [f"MAV channel {i+1}" for i in range(8)] + [f"WL channel {i+1}" for i in range(8)] + [f"RMS channel {i+1}" for i in range(8)] + [f"Maximum Amplitude channel {i+1}" for i in range(8)]
    #featuresNames = [f"CSP {i+1}" for i in range(8)]

    #bonesToTest = [i for i in range(17)]
    bonesToTest = [2, 4, 7, 10, 14]
    for b in bonesToTest:

        #csp = CSP(n_components=8, reg="oas", log=True)
        #x_csp = csp.fit_transform(np.array(x), np.array(y).transpose()[b].transpose() // 1)
        #x_new, selectedFeatures = MrmrRegression(np.array(x_csp).transpose(), np.array(y).transpose()[b:b+1], 2)
        x_new, selectedFeatures = MrmrRegression(np.array(x).transpose(), np.array(y).transpose()[b:b + 1], 2)

        plt.hist(np.array(y).transpose()[b], bins=100)
        plt.show()

        plt.hexbin(x_new[0], x_new[1], C=np.array(y).transpose()[b], cmap="seismic", gridsize=50, reduce_C_function=np.mean)
        plt.xlabel("First selected feature : " + featuresNames[selectedFeatures[0]])
        plt.ylabel("Second selected feature : " + featuresNames[selectedFeatures[1]])
        plt.title(f"Joint Angle {b}")
        plt.colorbar()
        plt.show()


        plt.hexbin(x_new[0], x_new[1], C=np.array(y).transpose()[b], cmap="turbo", gridsize=50, reduce_C_function=np.std)
        plt.xlabel("First selected feature : " + featuresNames[selectedFeatures[0]])
        plt.ylabel("Second selected feature : " + featuresNames[selectedFeatures[1]])
        plt.title(f"Joint Angle {b}")
        plt.colorbar()
        plt.show()




def regressionForPlotCSP(X, Y):
    set_log_level('warning')
    x = np.concatenate(X[0]).transpose((0, 2, 1))
    y = np.concatenate(Y[0])

    b1, b2 = 0, int(len(x) * 0.2)
    x_fold_train = np.concatenate((x[:b1], x[b2:]))
    x_fold_test = np.array(x[b1:b2])
    y_fold_train = np.concatenate((y[:b1], y[b2:]))
    y_fold_test = np.array(y[b1:b2])

    csp_train, csp_test = [], []
    for i in range(y_fold_test.shape[1]):
        if i != 13:
            csp = CSP(n_components=8, reg="oas", log=True)
            csp.fit(np.array(x_fold_train), np.array(y_fold_train).transpose()[i].transpose() // 10)
            csp_train.append(csp.transform(np.array(x_fold_train)))
            csp_test.append(csp.transform(np.array(x_fold_test)))
    x_fold_train = np.concatenate(np.array(csp_train), axis=1)
    x_fold_test = np.concatenate(np.array(csp_test), axis=1)

    model = MLPRegressor(hidden_layer_sizes=(100, 50), learning_rate="invscaling", learning_rate_init=0.001,
                         max_iter=200, epsilon=0.5, early_stopping=True, verbose=False, shuffle=False)
    model.fit(x_fold_train, y_fold_train)
    p = model.predict(x_fold_test)

    print(estimationScore(y_fold_test, p))


    p = p.transpose()
    y_fold_test = y_fold_test.transpose()

    for i in range(len(p)):
        plt.plot(y_fold_test[i][:4000] + i*150, c="blue")
        plt.plot(p[i][:4000] + i*150, c="orange")
    plt.show()



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
    #np.save(f"RegressionScoreStandard{testName}SingleSuject.npy", scores)



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
    np.save(f"RegressionScoreStandard{testName}InterSubjectsSessions.npy", scores)



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
    np.save(f"RegressionScoreStandard{testName}InterSubjectsSubjects.npy", scores)





if __name__ == "__main__":
    arg = sys.argv
    DATA_FOLDER_PATH = "C:/Users/marti/Desktop/memoire/data"
    if len(arg) < 2:
        # quit()
        pass
    else:
        DATA_FOLDER_PATH = arg[1]

    X, Y = loadData(DATA_FOLDER_PATH, (0, 1), ("freemove", 6), 1, "CSP")




    regressionTests(X, Y, testName="CSP", processData=computeCSP)

    #regressionForPlotCSP(X, Y)

