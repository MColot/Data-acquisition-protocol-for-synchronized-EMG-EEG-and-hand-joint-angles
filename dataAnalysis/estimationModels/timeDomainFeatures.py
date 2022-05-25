"""
Features extraction from epochs of EMG signal given in the time domain
"""
# ref: https://github.com/mohsenbme/EMG-Feature-extraction-and-evaluation

from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
from scipy.fft import fft

class EmgFeature(BaseEstimator, TransformerMixin):
    """
    Base class for the features extracted from EMG data
    """

    def __init__(self):
        pass

    def fit(self, X, y=None):
        """
        Do nothing. For compatibility purpose
        :param X: EMG signal in a ndarray of shape (n_trials, n_frames)
        :param y: labels of the trials in a ndarray of shape (n_trials,)
        :return: self
        """
        return self

    def transform(self, X):
        """
        gives the input signal in a numpy array
        :param X: EMG signal in a ndarray of shape (n_trials, n_frames)
        :return: EMG signal in a ndarray of shape (n_trials, n_frames)
        """
        if not isinstance(X, np.ndarray):
            X = np.array(X)
        return X


class MAV(EmgFeature):
    """
    Mean Absolute Value of the signal
    """

    def transform(self, X):
        """
        Estimate Mean Absolute Values
        :param X: EMG signal in a ndarray of shape (n_trials, n_frames)
        :return: MAV of X in a ndarray of shape (n_trials,)
        """
        X = super().transform(X)

        return np.mean(np.abs(X), 1)


class WL(EmgFeature):
    """
    Waveform Length of the signal
    """

    def transform(self, X):
        """
        Estimate Waveform Length
        :param X: EMG signal in a ndarray of shape (n_trials, n_frames)
        :return: WL of X in a ndarray of shape (n_trials,)
        """
        X = super().transform(X)
        return np.array([sum([abs(x[i + 1] - x[i]) for i in range(len(x) - 1)]) for x in X])


class RMS(EmgFeature):
    """
    Root Mean Square of the signal
    """

    def transform(self, X):
        """
        Estimate Root Mean Square
        :param X: EMG signal in a ndarray of shape (n_trials, n_frames)
        :return: RMS of X in a ndarray of shape (n_trials,)
        """
        X = super().transform(X)
        return np.array([(sum(x ** 2) / len(x)) ** 0.5 for x in X])


class ZC(EmgFeature):
    """
    Zero-crossing of the signal
    """

    def __init__(self, threshold):
        super().__init__()
        self.threshold = threshold

    def transform(self, X):
        """
        Estimate Zero-crossing
        :param X: EMG signal in a ndarray of shape (n_trials, n_frames)
        :return: ZC of X in a ndarray of shape (n_trials,)
        """
        X = super().transform(X)
        zc = np.array([0 for _ in range(len(X))])
        for t in range(len(X)):
            x = X[t]
            for i in range(len(x) - 1):
                if x[i] * x[i + 1] < 0 and abs(x[i] - x[i + 1]) >= self.threshold:
                    zc[t] += 1
        return zc


class WAMP(EmgFeature):
    """
    Wilson amplitude of the signal
    """

    def __init__(self, threshold):
        super().__init__()
        self.threshold = threshold

    def transform(self, X):
        """
        Estimate Wilson amplitude
        :param X: EMG signal in a ndarray of shape (n_trials, n_frames)
        :return: WAMP of X in a ndarray of shape (n_trials,)
        """
        X = super().transform(X)
        wamp = np.array([0 for _ in range(len(X))])
        for t in range(len(X)):
            x = X[t]
            for i in range(len(x) - 1):
                if abs(x[i + 1] - x[i]) >= self.threshold:
                    wamp[t] += 1
        return wamp


class SSC(EmgFeature):
    """
    Sign Slope Change of the signal
    """

    def __init__(self, threshold):
        super().__init__()
        self.threshold = threshold

    def transform(self, X):
        """
        Estimate Sign Slope Change
        :param X: EMG signal in a ndarray of shape (n_trials, n_frames)
        :return: SSC of X in a ndarray of shape (n_trials,)
        """
        X = super().transform(X)
        ssc = np.array([0 for _ in range(len(X))])
        for t in range(len(X)):
            x = X[t]
            for i in range(1, len(x) - 1):
                if abs(x[i + 1] - x[i]) * abs(x[i] - x[i - 1]) >= self.threshold:
                    ssc[t] += 1
        return ssc


class MaximumAbsoluteAmplitude(EmgFeature):
    """
    Maximum Absolute Amplitude of the signal
    """

    def transform(self, X):
        """
        Estimate maximum amplitude
        :param X: EMG signal in a ndarray of shape (n_trials, n_frames)
        :return: maximum amplitude of X in a ndarray of shape (n_trials,)
        """
        X = super().transform(X)
        return np.array([max(abs(x)) for x in X])


# https://www.sciencedirect.com/science/article/pii/S1746809420303426

class IEMG(EmgFeature):
    """
    Integral of the EMG
    """

    def transform(self, X):
        X = super().transform(X).transpose()
        res = 0
        for i in range(1, len(X) // 2):
            res += X[2 * i - 2] + 4 * X[2 * i - 1] + X[2 * i]
        return res / 3


class MM(EmgFeature):
    """
    Mathematical moment
    """

    def __init__(self, k):
        super().__init__()
        self.k = k

    def transform(self, X):
        X = super().transform(X).transpose()
        res = sum(X ** self.k)
        return res / (len(X) - 1)


class FFT(EmgFeature):
    """
    Fast Fourier Transform
    """

    def transform(self, X):
        X = super().transform(X)
        return fft(X)
