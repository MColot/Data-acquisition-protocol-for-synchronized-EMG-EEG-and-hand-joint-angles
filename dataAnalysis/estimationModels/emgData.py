import numpy as np
from estimationModels.signalTransformation import Envelope, Normalize, epochSegmentation, lowpassFilter, highpassFilter
import matplotlib.pyplot as plt


class EmgMocapDataSet:
    """
    Class containing a dataset of epochs of EMG data with corresponding timestamps, poses and bone rotations
    """

    def __init__(self, emg, mocap, mocapTimestamps, poses, windowSize=500, windowStep=500, lowpass=50, signalCut=(0, 0),
                 useEnvelope=True, discretise=None):
        """
        :param type: 1=position, 2=speed
        """
        dataEmg = np.array(emg)
        dataMocap = np.array(mocap)

        # enveloppes the signal
        if useEnvelope:
            processedData = np.array(
                [Envelope(emg, lowPass=lowpass, sfreq=2000, highBand=20, lowBand=500) for emg in dataEmg])
        else:
            processedData = np.array(
                [highpassFilter(lowpassFilter(emg, lowPass=lowpass, sfreq=2000), highpass=20, sfreq=2000) for emg in
                 dataEmg])
        processedData = np.array([emg for emg in processedData])
        processedData = np.array([Normalize(emg) for emg in processedData])
        self.__raw = processedData

        # cuts the signal in epochs
        eventTimes = [(i * windowStep, i * windowStep + windowSize) for i in range(int(signalCut[0] / windowStep), int(
            (len(dataEmg[0]) - signalCut[1] - windowSize) / windowStep))]
        epochs = np.array([epochSegmentation(emg, eventTimes) for emg in processedData]).transpose((1, 0, 2))
        self.__epochs = []
        self.__labels = []
        self.__poses = []
        j = 0
        for i in range(len(eventTimes)):
            while j < len(mocapTimestamps) - 1 and mocapTimestamps[j] < eventTimes[i][1] / 2:  # divide by 2 because emg is at 2000Hz and mocap timestamps given in ms
                j += 1
            if discretise is None:
                x = dataMocap[j]
            else:
                x = [0] * len(dataMocap[j])
                for l in range(len(dataMocap[j])):
                    k = 0
                    while k < len(discretise) and discretise[k] < dataMocap[j][l]:
                        k += 1
                    x[l] = discretise[k]
            if abs(mocapTimestamps[j] - eventTimes[i][
                1] / 2) <= 10:  # 20 ms de marge ce qui correspond à la demi période d'échantillonage de l'oculus Quest
                self.__labels.append(x)
                self.__epochs.append(epochs[i])
                self.__poses.append(poses[j])

        self.__labels = np.array(self.__labels)
        self.__epochs = np.array(self.__epochs)
        self.__poses = np.array(self.__poses)

        # normalization
        # self.__labels -= sum(self.__labels) / len(self.__labels)
        # self.__labels /= np.std(self.__labels)

    def channels(self):
        return self.__epochs.transpose((1, 0, 2))

    def epochs(self):
        return self.__epochs

    def labels(self):
        return self.__labels

    def poses(self):
        return self.__poses

    def raw(self):
        return self.__raw
