from scipy import signal
import numpy as np


def lowpassFilter(X, lowPass, sfreq=2000):
    lowPass = lowPass / sfreq
    b2, a2 = signal.butter(4, lowPass, btype='lowpass')
    return signal.filtfilt(b2, a2, X)

def highpassFilter(X, highpass, sfreq=2000):
    highpass = highpass / sfreq
    b2, a2 = signal.butter(4, highpass, btype='highpass')
    return signal.filtfilt(b2, a2, X)


def Envelope(X, lowPass=150, sfreq=2000, highBand=20, lowBand=500):
        """
        Envelope
        :param X: EMG signal in a ndarray of shape (n_frames,)
        :return: envelope of X in a ndarray of shape (n_trials,)
        """
        # normalize cut-off frequencies to sampling frequency
        highBand = highBand / (sfreq / 2)
        lowBand = lowBand / (sfreq / 2)
        # create the EMG bandpass filter
        b1, a1 = signal.butter(4, [highBand, lowBand], btype='bandpass')

        # filter the EMG signal forward and backward
        filteredX = signal.filtfilt(b1, a1, X)
        # rectify the EMG signal
        rectifiedX = abs(filteredX)
        # create the second lowpass filter apply it to the rectified signal to obtain EMG envelope
        lowPass = lowPass / sfreq
        b2, a2 = signal.butter(4, lowPass, btype='lowpass')
        return signal.filtfilt(b2, a2, rectifiedX)


def Normalize(X):
        """
        :param X: EMG signal in a ndarray of shape (n_frames,)
        :return: normalization of X in a ndarray of shape (n_trials,)
        """
        # find mean and standard deviation of the signal
        N = len(X)
        mean = sum(X)/N
        sd = ((sum(X-mean)**2)/N)**0.5
        # subtract mean and divide signal by standard deviation
        normalized = []
        if sd == 0:
            return X - mean
        for x in X:
            normalized.append((x-mean)/sd)
        return np.array(normalized)


def normalizePSD(X):
    """
    divides all the values in a PSD by the maximum value of each frequency band
    """
    XasVect = np.array(X).flatten()
    maxVal = max(XasVect)
    return np.array(X)/maxVal if maxVal > 0 else X


def epochSegmentation(X, events):
    """
    separates the signal in epoch given the list of begin and end time of each epoch
    :param X: signal
    :param events: list of shape (nEpoch, 2) giving, for each epoch: the start frame number and the end frame number
    :return an array of epochs
    """
    epochs = []
    for i in range(len(events)):
        epochs.append(np.array([X[i] for i in range(events[i][0], events[i][1])]))
    return np.array(epochs)

