import numpy as np


def MrmrRegression(X, y, n):
    """
    Maximum relevancy, minimum redundancy feature selection
    :param X: feature set : shape = (n_features, n_samples)
    :param y: output : shape = (n_targets, n_samples)
    :param n: number of features to extract
    """
    selectedFeatures = []

    for s in range(n):
        print(f"MRMR : finding feature {s+1} on {n}")
        cor = [0] * len(X)
        for j in range(len(X)):
            x = X[j]
            for k in range(len(y)):
                c = abs(np.corrcoef(x=x, y=y[k])[0, 1]) / len(y)
                if not np.isnan(c):
                    cor[j] += c
            for k in range(len(selectedFeatures)):
                c = abs(np.corrcoef(x=x, y=X[selectedFeatures[k]])[0, 1]) / len(selectedFeatures)
                if not np.isnan(c):
                    cor[j] -= c
        XwithCor = [(X[i], 0 if np.isnan(cor[i]) else cor[i], i) for i in range(len(X))]
        XwithCor = sorted(XwithCor, reverse=True, key=lambda x: x[1])
        i = 0
        while XwithCor[i][2] in selectedFeatures:
            i += 1
        selectedFeatures.append(XwithCor[i][2])
    print(selectedFeatures)
    selectedX = []
    for elem in selectedFeatures:
        selectedX.append(X[elem])
    return np.array(selectedX), selectedFeatures
