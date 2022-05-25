import numpy as np


def MSE(y, y_hat):
    """
    Computes mean square error
    :param y: true value
    :param y_hat: estimated value
    :return: mse
    """
    y = np.array(y)
    y_hat = np.array(y_hat)
    return sum((y_hat-y)**2)/len(y)


def NMSE(y, y_hat):
    """
    comptues normalized mean square error
    :param y: true value
    :param y_hat: estimated value
    :return: nmse
    """
    #remove bone 13 as it has always the same value
    y = np.concatenate((y.transpose()[:13], y.transpose()[14:])).transpose()
    y_hat = np.concatenate((y_hat.transpose()[:13], y_hat.transpose()[14:])).transpose()

    mse = MSE(y, y_hat)
    sd = MSE(y, [sum(y)/len(y)]*len(y))
    return sum(mse / sd)/len(mse)


def truncate(num, n):
    """
    returns num truncated after the n-th decimal
    """
    integer = int(num * (10 ** n)) / (10 ** n)
    return float(integer)


def discretise(serie, bins):
    """
    :param serie: list to discretise
    :param bins: list of bins of the form (n_bins, 2) where the 1st elem of each bin is the start of the bin and the 2nd is the end
    """
    res = []
    for elem in serie:
        for i in range(len(bins)):
            if bins[i][0] <= elem < bins[i][1]:
                res.append(i)
    return res


def readCsv(path):
    """
    reads a csv file and place its content in a matrix
    :param path: path to the csv file
    :return: the matrix
    """
    res = []
    with open(path, "r") as f:
        for line in f:
            elem = line.split(";")
            res.append(elem)
    return res


def stringToVector(s):
    """
    takes a string in format 'a,b,c' with any amount of number and returns the vector that is represented
    """
    values = s.split(",")
    return tuple(float(v) for v in values)

















