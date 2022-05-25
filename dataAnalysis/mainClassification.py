import sys

import numpy as np
import ezc3d
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn import preprocessing
from sklearn.metrics import plot_confusion_matrix
from preprocessing.c3dRecovery import recoverFile

from estimationModels.timeDomainFeatures import MAV, WL, RMS, MaximumAbsoluteAmplitude
from estimationModels.dataGenerators import createDatasetLabels
from estimationModels.signalTransformation import lowpassFilter, highpassFilter
from estimationModels.featureSelection import MrmrRegression
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_classif
from sklearn.pipeline import Pipeline
from mne.decoding import CSP
from estimationModels.dataLoader import loadData
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import roc_auc_score
from itertools import cycle
from sklearn.metrics import RocCurveDisplay, accuracy_score
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
import seaborn as sn
import pandas as pd
from estimationModels.featureSelection import MrmrRegression

#roc curves : https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html
def GenerateRocCurve(x, y, model, title="", processData=lambda xtrain, xtest, ytrain: (xtrain, xtest)):
    x = np.array(x)
    y = np.array(y)
    cv = StratifiedKFold(n_splits=5)
    classifier = OneVsRestClassifier(model)
    y_binary = label_binarize(y, classes=["2", "7", "19", "23"]).transpose()

    for j in range(4):
        tprs = []
        aucs = []
        mean_fpr = np.linspace(0, 1, 100)
        fig, ax = plt.subplots()
        for i, (train, test) in enumerate(cv.split(x, y_binary[j])):

            x_train, x_test = processData(x[train], x[test], y[train])

            classifier.fit(x_train, y_binary[j][train])
            viz = RocCurveDisplay.from_estimator(
                classifier,
                x_test,
                y_binary[j][test],
                name="ROC fold {}".format(i),
                alpha=0.3,
                lw=1,
                ax=ax,
            )
            interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
            interp_tpr[0] = 0.0
            tprs.append(interp_tpr)
            aucs.append(viz.roc_auc)

        ax.plot([0, 1], [0, 1], linestyle="--", lw=2, color="r", label="Chance", alpha=0.8)

        mean_tpr = np.mean(tprs, axis=0)
        mean_tpr[-1] = 1.0
        mean_auc = auc(mean_fpr, mean_tpr)
        std_auc = np.std(aucs)
        ax.plot(
            mean_fpr,
            mean_tpr,
            color="b",
            label=r"Mean ROC (AUC = %0.2f $\pm$ %0.2f)" % (mean_auc, std_auc),
            lw=2,
            alpha=0.8,
        )

        std_tpr = np.std(tprs, axis=0)
        tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
        tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
        ax.fill_between(
            mean_fpr,
            tprs_lower,
            tprs_upper,
            color="grey",
            alpha=0.2,
            label=r"$\pm$ 1 std. dev.",
        )

        ax.set(
            xlim=[-0.05, 1.05],
            ylim=[-0.05, 1.05],
            title=f"ROC curve of class {j}, 5-fold, {title}",
        )
        ax.legend(loc="lower right")
        plt.show()








def ComputeErrorsDurations(p, y):
    curr = 0
    errors = []
    for i in range(1, len(y)):
        if y[i] == y[i-1] and y[i] != p[i]:
            curr+=1
        elif curr > 0:
            errors.append(curr)
            curr=0
    return errors

def plotErrorDurations(errorDurations):
    models = "Random Forest", "KNN", "Logistic Regression", "LDA", "MLP"
    maxValueAllModel = 0
    for m in range(5):
        maxValue = max(errorDurations[m])
        maxValueAllModel = max(maxValueAllModel, maxValue)
        counts = [0] * maxValue
        for elem in errorDurations[m]:
            counts[min(maxValue, elem) - 1] += 1
        for n in range(maxValue):
            counts[n] /= len(errorDurations[m])
        plt.bar(np.array(range(1, maxValue+1)) + (m - 1.5) * 0.15, counts, width=0.15, label=models[m])
    plt.title("Error duration in real-time classification")
    plt.legend()
    plt.xlabel("Duration (frames count)")
    plt.ylabel("Frequency")
    plt.xticks(range(1, maxValueAllModel+1), range(1, maxValueAllModel+1))
    plt.show()


def computeCSP(x_train, x_test, y_train):
    x_train = np.array(x_train).transpose((0, 2, 1))
    x_test = np.array(x_test).transpose((0, 2, 1))

    csp = CSP(n_components=8, reg="oas", log=True)
    csp.fit(np.array(x_train), np.array(y_train))

    x_fold_train = csp.transform(np.array(x_train))
    x_fold_test = csp.transform(np.array(x_test))
    return x_fold_train, x_fold_test




def printConfusionMatrix(m, title=""):
    for j in range(len(m[0])):
        tot = 0
        for i in range(len(m)):
            tot+=m[i][j]
        for i in range(len(m)):
            m[i][j]/=tot


    plt.rcParams["figure.figsize"] = (7,6)
    df_cm = pd.DataFrame(m, range(m.shape[0]), range(m.shape[1]))
    # plt.figure(figsize=(10,7))
    sn.set(font_scale=1.4)  # for label size
    sn.heatmap(df_cm, annot=True, annot_kws={"size": 16}, cmap="coolwarm_r", vmin=0, vmax=1)  # font size
    plt.title(f"Confusion Matrix : {title}")
    plt.xlabel("Real class")
    plt.ylabel("Predicted class")
    plt.show()



def classificationTests(X, Y, processData=lambda xtrain, xtest, ytrain: (xtrain, xtest)):
    modelsNames = ["Random forest", "KNN", "Logistic regression", "LDA", "MLP"]
    print("----")
    print("Intra subject estimation")
    meanScores = np.zeros(5)
    errorDurations = [[], [], [], [], []]
    confusionMatrices = [np.zeros((4, 4)) for model in range(5)]
    for i in range(len(X)):
        print("    estimation for", i)
        x = np.concatenate(X[i])
        y = np.concatenate(Y[i])

        #GenerateRocCurve(x, y, LogisticRegression(max_iter=1000), "Single Subject", processData=processData)

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
                cm = confusion_matrix(y_fold_test, model.predict(x_fold_test))
                confusionMatrices[m] += cm
                errorDurations[m].extend(ComputeErrorsDurations(model.predict(x_fold_test), y_fold_test))

            crossValScores += np.array(([np.mean(s) for s in scores]))

        print("scores subject : ", crossValScores/5)

        meanScores += crossValScores/5
    print("    Mean Scores")
    print(meanScores / len(X))
    #plotErrorDurations(errorDurations)

    #for m in range(5):
    #    printConfusionMatrix(confusionMatrices[m], modelsNames[m] + " - Single Subject")



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
    GenerateRocCurve(x, y, LogisticRegression(max_iter=1000), "Inter Subject - Session", processData=processData)

    crossValScores = np.zeros(5)
    errorDurations = [[], [], [], [], []]
    confusionMatrices = [np.zeros((4, 4)) for model in range(5)]
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
            cm = confusion_matrix(y_fold_test, model.predict(x_fold_test))
            confusionMatrices[m] += cm
            errorDurations[m].extend(ComputeErrorsDurations(model.predict(x_fold_test), y_fold_test))

        crossValScores += np.array(([np.mean(s) for s in scores]))
    print("scores Inter subject 1 : ", crossValScores / 5)
    #plotErrorDurations(errorDurations)

    for m in range(5):
        printConfusionMatrix(confusionMatrices[m], modelsNames[m] + " - Inter Subject (session)")

    print("----")
    print("Inter subject estimation 2 (isolate subject)")
    x = []
    y = []
    for i in range(len(X)):
        x.extend(np.concatenate(X[i]))
        y.extend(np.concatenate(Y[i]))

    x = np.array(x)
    #GenerateRocCurve(x, y, LogisticRegression(max_iter=1000), "Inter Subject - Subject", processData=processData)

    crossValScores = np.zeros(5)
    errorDurations = [[], [], [], [], []]
    confusionMatrices = [np.zeros((4, 4)) for model in range(5)]
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
            cm = confusion_matrix(y_fold_test, model.predict(x_fold_test))
            confusionMatrices[m] += cm
            errorDurations[m].extend(ComputeErrorsDurations(model.predict(x_fold_test), y_fold_test))

        crossValScores += np.array(([np.mean(s) for s in scores]))
    print("scores Inter subject 2 : ", crossValScores / 5)
    #plotErrorDurations(errorDurations)

    #for m in range(5):
    #    printConfusionMatrix(confusionMatrices[m], modelsNames[m] + " - Inter Subject (subject)")





def handsTests(XR, XL, YR, YL):
    print("----")
    print("Intra subject estimation")
    meanScores = np.zeros(5)
    for i in range(len(XR)):
        size = min(sum([len(e) for e in YR[i]]), sum([len(e) for e in YL[i]]))
        print(size)
        print("    estimation for", i)
        scoreRF_right = cross_val_score(RandomForestClassifier(n_estimators=10), np.concatenate(XR[i])[:size], np.concatenate(YR[i])[:size], cv=5)
        print(f"score random forest, Right hand only : {np.mean(scoreRF_right) * 100}%")

        scoreRF_left = cross_val_score(RandomForestClassifier(n_estimators=10), np.concatenate(XL[i])[:size], np.concatenate(YL[i])[:size], cv=5)
        print(f"score random forest, Left hand only : {np.mean(scoreRF_left) * 100}%")


        mixedX = []
        mixedY = []
        for j in range(5):
            if j < len(XR[i]):
                mixedX.append(XR[i][j])
                mixedY.append(YR[i][j])
            if j < len(XL[i]):
                mixedX.append(XL[i][j])
                mixedY.append(YL[i][j])
        mixedX = np.concatenate(mixedX)
        mixedY = np.concatenate(mixedY)

        scoreRF_mixed = cross_val_score(RandomForestClassifier(n_estimators=10), mixedX[:size], mixedY[:size], cv=5)
        print(f"score random forest, 2 hands : {np.mean(scoreRF_mixed) * 100}%")

        scoreRF_mixed_moreData = cross_val_score(RandomForestClassifier(n_estimators=10), mixedX, mixedY, cv=5)
        print(f"score random forest, 2 hands with more data: {np.mean(scoreRF_mixed_moreData) * 100}%")

        scoreRF_separated = cross_val_score(RandomForestClassifier(n_estimators=10), np.concatenate([np.concatenate(XR[i])[:int(size*4/5)], np.concatenate(XL[i])[:int(size*4/5)]]), np.concatenate([np.concatenate(YR[i])[:int(size*4/5)], np.concatenate(YL[i])[:int(size*4/5)]]), cv=2)
        print(f"score random forest, train one, test the other : {np.mean(scoreRF_separated) * 100}%")


        meanScores += np.array((np.mean(scoreRF_right), np.mean(scoreRF_left), np.mean(scoreRF_mixed), np.mean(scoreRF_mixed_moreData), np.mean(scoreRF_separated)))
    print("    Mean Scores")
    print(meanScores / len(XR))


import matplotlib.pyplot as plt
def featureAnalysisPCA(x, y):

    """pca = PCA(n_components=2)
    pca = pca.fit(x)
    print(pca.explained_variance_ratio_)
    x = pca.transform(x)"""

    bestFeatures = SelectKBest(mutual_info_classif, k=2)
    bestFeatures.fit(x, y)
    print(bestFeatures.get_support())
    x = bestFeatures.transform(x)



    """dataDict = {"2":[[] for _ in range(2)], "7":[[]for _ in range(2)], "19":[[]for _ in range(2)], "23":[[]for _ in range(2)]}
    for j in range(len(x)):
        for k in range(2):
            dataDict[y[j]][k].append(x[j][k])


    for key in dataDict:
        plt.scatter(dataDict[key][0], dataDict[key][1], label=key, s=2)"""

    plt.scatter(x.transpose()[0], x.transpose()[1], c = [["tab:blue", "tab:orange", "tab:green", "tab:red"][["2", "7", "19", "23"].index(e)] for e in y], s=2)


    plt.xlabel("First principal component")
    plt.ylabel("Second principal component")
    plt.show()









def featureImportanceRF(X, Y, processData=lambda xtrain, xtest, ytrain: (xtrain, xtest)):
    print("----")
    print("Intra subject estimation")
    importances = []
    for i in range(len(X)):
        print("    estimation for", i)
        x = np.concatenate(X[i])
        y = np.concatenate(Y[i])


        for k in range(5):
            b1, b2 = int(len(x) * 0.2 * k), int(len(x) * 0.2 * (k + 1))
            x_fold_train = np.concatenate((x[:b1], x[b2:]))
            x_fold_test = np.array(x[b1:b2])
            y_fold_train = np.concatenate((y[:b1], y[b2:]))

            x_fold_train, x_fold_test = processData(x_fold_train, x_fold_test, y_fold_train)

            # model = RandomForestClassifier(n_estimators=30)
            model = RandomForestRegressor(n_estimators=30)
            model.fit(x_fold_train, y_fold_train)

            importances.append(model.feature_importances_)

    print(np.mean(importances, axis=(0,)))

    return

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

    importances = []
    for k in range(5):
        b1, b2 = int(len(x) * 0.2 * k), int(len(x) * 0.2 * (k + 1))
        x_fold_train = np.concatenate((x[:b1], x[b2:]))
        x_fold_test = np.array(x[b1:b2])
        y_fold_train = np.concatenate((y[:b1], y[b2:]))

        x_fold_train, x_fold_test = processData(x_fold_train, x_fold_test, y_fold_train)

        #model = RandomForestClassifier(n_estimators=30)
        model = RandomForestRegressor(n_estimators=30)
        model.fit(x_fold_train, y_fold_train)
        importances.append(model.feature_importances_)
    print(np.mean(importances, axis=(0,)))

    importances = []
    print("----")
    print("Inter subject estimation 2 (isolate subject)")
    x = []
    y = []
    for i in range(len(X)):
        x.extend(np.concatenate(X[i]))
        y.extend(np.concatenate(Y[i]))

    x = np.array(x)

    for k in range(5):
        b1, b2 = int(len(x) * 0.2 * k), int(len(x) * 0.2 * (k + 1))
        x_fold_train = np.concatenate((x[:b1], x[b2:]))
        x_fold_test = np.array(x[b1:b2])
        y_fold_train = np.concatenate((y[:b1], y[b2:]))

        x_fold_train, x_fold_test = processData(x_fold_train, x_fold_test, y_fold_train)

        # model = RandomForestClassifier(n_estimators=30)
        model = RandomForestRegressor(n_estimators=30)
        model.fit(x_fold_train, y_fold_train)
        importances.append(model.feature_importances_)
    print(np.mean(importances, axis=(0,)))





CSP_FIT_DATA_STEP = 10
def computeCSPregression(x_train, x_test, y_train):
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






def MrmrFeatureImportance(X, Y):
    allSelectedFeatures = []

    for i in range(len(X)):
        y = np.array(np.concatenate(Y[i]))
        x = np.array(np.concatenate(X[i]))

        x, a = computeCSPregression(x, x, y)


        x_new, selectedFeatures = MrmrRegression(x.transpose(), y.transpose(), 128)
        allSelectedFeatures.append(selectedFeatures)

    print(allSelectedFeatures)




if __name__ == "__main__":
    arg = sys.argv
    DATA_FOLDER_PATH = "C:/Users/marti/Desktop/memoire/data"
    if len(arg) < 2:
        #quit()
        pass
    else:
        DATA_FOLDER_PATH = arg[1]

    print("TESTS using TDF")
    X, Y = loadData(DATA_FOLDER_PATH, (0, 1), datasetType=("freemove", 1), dataType="CSP", labelType=1)

    MrmrFeatureImportance(X, Y)
    #featureImportanceRF(X, Y, processData=computeCSPregression)


    #classificationTests(X, Y)

    #print("TESTS using CSP")
    #X, Y = loadData(DATA_FOLDER_PATH, (0, 1), datasetType=("sign", 5), dataType="CSP")
    #classificationTests(X, Y, processData=computeCSP)
    #featureAnalysisPCA(X, Y)


    """csp = CSP(n_components=8, reg="oas", log=True)
    x, y = [], []
    for i in range(len(X)):
        x.extend(np.concatenate(X[i]).transpose((0, 2, 1)))
        y.extend(np.concatenate(Y[i]))
    x = np.array(x)
    y = np.array(y)
    x = csp.fit_transform(x, y)
    featureAnalysisPCA(x, y)"""

    #XR, YR = loadData(DATA_FOLDER_PATH, (0,))
    #XL, YL = loadData(DATA_FOLDER_PATH, (1,))
    #handsTests(XR, XL, YR, YL)


