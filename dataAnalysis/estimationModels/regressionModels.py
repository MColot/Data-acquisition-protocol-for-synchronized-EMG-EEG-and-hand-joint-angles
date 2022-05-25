"""
Sets of regression models to predict the motion capture signal from the EMG signal
"""

from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer
from sklearn.utils import shuffle
import numpy as np
import matplotlib.pyplot as plt
from estimationModels.utils import NMSE


def normalized_mean_squared_error(estimator, x, y):
    estimator.fit(x, y)
    pred = estimator.predict(x)
    return NMSE(y, pred)


class RegressionModel:
    def __init__(self, X, y, model, name="model"):
        self.X = X
        self.y = y
        self.model = model
        self.name = name

    def crossValMSE(self, k=4):
        """
        :param k: number of folds
        :return: Mean square error of each fold of the cross validation
        """
        mse = -cross_val_score(self.model, self.X, self.y, cv=k, scoring='neg_mean_squared_error')
        return mse

    def crossValNMSE(self, k=4):
        """
        :param k: number of folds
        :return: Normalized Mean square error of each fold of the cross validation
        """
        nmseScorer = make_scorer(NMSE, greater_is_better=False)
        nmse = -cross_val_score(self.model, self.X, self.y, cv=k, scoring=nmseScorer)
        return nmse

    def crossValDefault(self, k=4):
        """
        cross validation using mse
        :param k: number of folds
        :return: mean score of the estimation
        """
        score = cross_val_score(self.model, self.X, self.y, cv=k)
        return score

    def makePrediction(self, plot=False):
        """
        cuts the data set into training and testing set
        trains the model on the training set
        tests the model on the testing set
        returns the prediction together with the actual data
        :param plot: tells to plot the result
        """
        X_train = shuffle(np.array(self.X[:int(len(self.X) * 0.75)]), random_state=123)
        X_test = np.array(self.X[int(len(self.X) * 0.75):])
        y_train = shuffle(np.array(self.y[:int(len(self.y) * 0.75)]), random_state=123)
        y_test = np.array(self.y[int(len(self.y) * 0.75):])

        self.model.fit(X_train, y_train)
        pred = self.model.predict(X_test)

        if plot:
            if(len(y_test.shape) == 2):
                y_test = y_test.transpose()
                pred = np.array(pred).transpose()
                for i in range(len(pred)):
                    plt.plot(y_test[i] +i*150, color="blue")
                    plt.plot(pred[i] +i*150, color="orange")
            else:
                plt.plot(y_test, color="blue")
                plt.plot(pred, color="orange")
            plt.title(self.name)
            plt.xlabel("time")
            plt.ylabel("position")
            plt.show()

        return y_test, pred

