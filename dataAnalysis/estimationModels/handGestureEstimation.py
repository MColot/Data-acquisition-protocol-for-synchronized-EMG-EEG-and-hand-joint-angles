# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, RandomForestClassifier, \
    GradientBoostingClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.svm import SVR
from sklearn.multioutput import MultiOutputRegressor
from estimationModels.regressionModels import RegressionModel
from estimationModels.timeDomainFeatures import *

"""
Set of functions to test different regression and estimation models with a common interface
"""
FOLD_COUNT = 4

def testLogRegClass(X, y, inputType=""):
    """
    y should be discretised here
    """
    # logReg = RegressionModel(X, y, MultiOutputClassifier(LogisticRegression(max_iter=5000)), f"Logistic regression on {inputType}")
    logReg = RegressionModel(X, y, LogisticRegression(max_iter=10000), f"Logistic regression on {inputType}")
    print(f"score of log reg Model: {sum(logReg.crossValDefault(FOLD_COUNT)) / FOLD_COUNT}")
    return logReg.makePrediction(plot=True)


def testRandomForestClass(X, y, inputType=""):
    """
    y should be discretised here
    """
    randomForestClass = RegressionModel(X, y, RandomForestClassifier(n_estimators=100), f"Random forest on {inputType}")
    print(f"score of random forest Model: {sum(randomForestClass.crossValDefault(FOLD_COUNT)) / FOLD_COUNT}")
    return randomForestClass.makePrediction(plot=True)


def testKnnClass(X, y, inputType="", k=10):
    """
    y should be discretised here
    """
    knn = RegressionModel(X, y, KNeighborsClassifier(n_neighbors=k), f"KNN on {inputType}")
    print(f"score of KNN Model: {sum(knn.crossValDefault(FOLD_COUNT)) / FOLD_COUNT}")
    return knn.makePrediction(plot=True)


def testGradBoostClass(X, y, inputType="", k=10):
    """
    y should be discretised here
    """
    gb = RegressionModel(X, y, GradientBoostingClassifier(n_estimators=k), f"Gradiant Boosting on {inputType}")
    print(f"score of Gradiant boosting Model: {sum(gb.crossValDefault(FOLD_COUNT)) / FOLD_COUNT}")
    return gb.makePrediction(plot=True)


def testLinReg(X, y, inputType=""):
    linReg = RegressionModel(X, y, LinearRegression(), f"Linear regression on {inputType}")
    print(f"nmse linear Model: {sum(linReg.crossValNMSE(FOLD_COUNT)) / FOLD_COUNT}")
    return linReg.makePrediction(plot=True)


def testRandomForest(X, y, inputType="", nbTree=100):
    randomForestReg = RegressionModel(X, y, RandomForestRegressor(n_estimators=nbTree), f"Random forest on {inputType}")
    print(f"nmse random forest Model: {sum(randomForestReg.crossValNMSE(FOLD_COUNT)) / FOLD_COUNT}")
    return randomForestReg.makePrediction(plot=True)


def testKNN(X, y, inputType=""):
    knnReg = RegressionModel(X, y, KNeighborsRegressor(n_neighbors=5), f"KNN on {inputType}")
    print(f"nmse knn Model: {sum(knnReg.crossValNMSE(FOLD_COUNT)) / FOLD_COUNT}")
    return knnReg.makePrediction(plot=True)


def testGradBoost(X, y, inputType=""):
    gradientBoostReg = RegressionModel(X, y, MultiOutputRegressor(GradientBoostingRegressor(n_estimators=10)),
                                       f"Gradient boosting on {inputType}")
    print(f"nmse gradiant boosting Model: {sum(gradientBoostReg.crossValNMSE(FOLD_COUNT)) / FOLD_COUNT}")
    return gradientBoostReg.makePrediction(plot=True)


def testModels(X, y, inputType=""):
    linReg = RegressionModel(X, y, LinearRegression(), f"Linear regression on {inputType}")
    knnReg = RegressionModel(X, y, KNeighborsRegressor(n_neighbors=5), f"KNN on {inputType}")
    randomForestReg = RegressionModel(X, y, RandomForestRegressor(n_estimators=20), f"Random forest on {inputType}")
    gradientBoostReg = RegressionModel(X, y, MultiOutputRegressor(GradientBoostingRegressor(n_estimators=20)),
                                       f"Gradient boosting on {inputType}")
    svrReg = RegressionModel(X, y, MultiOutputRegressor(SVR(kernel="poly")), f"SVR on {inputType}")

    print(f"nmse linear Model: {sum(linReg.crossValNMSE(FOLD_COUNT)) / FOLD_COUNT}")
    linReg.makePrediction(plot=True)

    print(f"nmse knn Model: {sum(knnReg.crossValNMSE(FOLD_COUNT)) / FOLD_COUNT}")
    knnReg.makePrediction(plot=True)

    print(f"nmse random forest Model: {sum(randomForestReg.crossValNMSE(FOLD_COUNT)) / FOLD_COUNT}")
    randomForestReg.makePrediction(plot=True)

    print(f"nmse svr Model: {sum(svrReg.crossValNMSE(FOLD_COUNT)) / FOLD_COUNT}")
    svrReg.makePrediction(plot=True)

    print(f"nmse gradient boosting Model: {sum(gradientBoostReg.crossValNMSE(FOLD_COUNT)) / FOLD_COUNT}")
    gradientBoostReg.makePrediction(plot=True)