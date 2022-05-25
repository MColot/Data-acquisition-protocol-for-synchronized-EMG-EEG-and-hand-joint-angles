import numpy as np
from estimationModels.emgData import EmgDataSetRegression
import os, sys
from scipy.signal import spectrogram
from estimationModels.timeDomainFeatures import MAV, WL, RMS, MaximumAbsoluteAmplitude, MM
from estimationModels.utils import truncate, discretise
from pyriemann.estimation import XdawnCovariances
from pyriemann.tangentspace import TangentSpace
from mne.decoding import CSP


def savePoses(dataEnveloppe, outPath):
    """
    saves the poses in a numpy matrix
    """
    for i in range(len(dataEnveloppe)):
        np.save(f"{outPath}/{datasetNames[i]}_poses.npy", dataEnveloppe[i].poses())


def saveRotations(dataEnveloppe, outPath):
    """
    saves the bones rotations in a numpy matrix
    """
    for i in range(len(dataEnveloppe)):
        np.save(f"{outPath}/{datasetNames[i]}_y.npy", dataEnveloppe[i].labels())


def saveSpectrograms(data, outPath):
    """
    computes the spectrogram and saves it in a numpy matrix
    """
    for i in range(len(data)):
        spect = [spectrogram(elem, fs=2000)[2] for elem in data[i].epochs()]
        bandSize = 20
        spectBands = [[[[sum(t[i * bandSize:(i + 1) * bandSize]) / bandSize for i in range(int(len(t) / bandSize))] for
                        t in np.array(emg).transpose()] for emg in e] for e in spect]
        X = np.array([np.concatenate(e) for e in spectBands])

        np.save(f"{outPath}/{datasetNames[i]}_spectrograms_X.npy", X)


def saveFilteredSignal(dataEnveloppe, outPath):
    """
    saves the whole signal in a numpy matrix
    """
    for i in range(len(dataEnveloppe)):
        np.save(f"{outPath}/{datasetNames[i]}_filtered150Hz_X.npy",
                [[emg[::] for emg in epoch] for epoch in dataEnveloppe[i].epochs()])


def saveTimeDomainFeatures(data, outPath):
    """
    computes time domains features and saves them in a numpy matrix
    """
    for i in range(len(data)):
        channels = data[i].epochs().transpose((1, 0, 2))
        mav = np.array([MAV().transform(emg) for emg in channels])  # ++
        wl = np.array([WL().transform(emg) for emg in channels])  # +
        rms = np.array([RMS().transform(emg) for emg in channels])  # ++
        # zc = np.array([ZC(0.00001).transform(emg) for emg in channels]) #--
        # wamp = np.array([WAMP(0.00005).transform(emg) for emg in channels]) #--
        # ssc = np.array([SSC(0.000000002).transform(emg) for emg in channels]) #--
        maxAmp = np.array([MaximumAbsoluteAmplitude().transform(emg) for emg in channels])  # +
        # iemg = np.array([IEMG().transform(emg) for emg in channels]) #--
        mm2 = np.array([MM(2).transform(emg) for emg in channels])  # ++
        # mm3 = np.array([MM(3).transform(emg) for emg in channels]) #--
        X = np.concatenate((mav, wl, rms, maxAmp, mm2)).transpose()

        np.save(f"{outPath}/{datasetNames[i]}_timeDomainFeatures_X.npy", X)


def saveXdawnCovMat(dataEnveloppe, outPath, saveCSP=False):
    """
    computes the Xdawn covariance matrices and saves it in a numpy matrix
    :param saveCSP: tells to also compute common spatial patterns and to save them in an other numpy matrix
    """
    for i in range(len(dataEnveloppe)):
        print(f"computing covariance matrices for dataset {i + 1} on {len(dataEnveloppe)}")

        posesToRecognize = ("6", "7", "15", "17", "19", "12")  # 12 is the resting pose

        y_poses_processed = []
        for j in range(len(dataEnveloppe[i].poses())):
            p = dataEnveloppe[i].poses()[j]
            p = str(p).replace(" ", "")
            if p in posesToRecognize:
                y_poses_processed.append(posesToRecognize.index(p))
            else:
                y_poses_processed.append(-1)

        covarianceMatrices = XdawnCovariances(3, estimator='oas', xdawn_estimator='oas')
        ts = TangentSpace()

        fittedCovMat = covarianceMatrices.fit(dataEnveloppe[i].epochs(), y_poses_processed)
        covMatX = fittedCovMat.transform(dataEnveloppe[i].epochs())
        fittedTs = ts.fit(covMatX, y_poses_processed)
        tsX = fittedTs.transform(covMatX)

        X = tsX
        np.save(f"{outPath}/{datasetNames[i]}_XdawnCovMatPose_X.npy", X)

        if saveCSP:
            csp = CSP(n_components=4, reg="oas", log=True)

            X = csp.fit_transform(covMatX, y_poses_processed)
            np.save(f"{outPath}/{datasetNames[i]}_cspCovMatPoses_X.npy", X)


if __name__ == "__main__":
    arg = sys.argv
    if len(arg) == 1:
        quit()
    pathToDatasets = arg[1]
    outPath = pathToDatasets + "/features"
    if not os.path.exists(outPath):
        os.mkdir(outPath)

    datasets = {}

    for (dirPath, dirNames, fileNames) in os.walk(pathToDatasets):
        for file in fileNames:
            if file[-4:] == ".npy":
                name = file.split("_")[0]
                type = file.split("_")[1][:-4]
                if not name in datasets:
                    datasets[name] = {}
                datasets[name][type] = np.load(f"{dirPath}/{file}")

    dataEnveloppe = []
    data = []
    datasetNames = []

    for datasetName in datasets:
        print(datasetName)
        dataset = datasets[datasetName]
        datasetNames.append(datasetName)
        if "EMG" in dataset.keys() and "mocapFrames" in dataset.keys() and "mocapTimestamps" in dataset.keys():
            dataEnveloppe.append(
                EmgDataSetRegression(dataset["EMG"], dataset["mocapFrames"], dataset["mocapTimestamps"],
                                     dataset["mocapPoses"], windowSize=600, windowStep=100, signalCut=(30000, 30000),
                                     lowpass=150, useEnvelope=True))
            # data.append(EmgDataSetRegression(dataset["EMG"], dataset["mocapFrames"], dataset["mocapTimestamps"], dataset["mocapPoses"], windowSize=600, windowStep=100, signalCut=(30000, 30000), lowpass=150, useEnvelope=False))


    saveRotations(dataEnveloppe, outPath)
    # savePoses(dataEnveloppe, outPath)

    saveFilteredSignal(dataEnveloppe, outPath)
    # saveTimeDomainFeatures(data, outPath)
    # saveSpectrograms(data, outPath)
    # saveXdawnCovMat(data, outPath, True)
