import os
import sys

import ezc3d
import matplotlib.pyplot as plt
import numpy as np

from estimationModels.emgData import EmgMocapDataSet
from estimationModels.handGestureEstimation import testRandomForestClass
from estimationModels.timeDomainFeatures import MAV, WL, RMS, MaximumAbsoluteAmplitude
from preprocessing.QuestDataProcessing import extractMocap
from preprocessing.c3dRecovery import recoverFile


def checkMocap(path):
    """
    Loads a motion capture csv file and plots its relevant data to visually check if it was correctly recorded
    :param path: path to the csv motion capture file
    :returns the motion capture data
    """
    timestampsLeft, framesLeft, posesLeft, wristLeft = extractMocap(path, 0)
    timestampsRight, framesRight, posesRight, WristRight = extractMocap(path, 1)

    print("Motion Capture ------------------------------")
    print(f"{len(timestampsLeft)} frames recorded")
    print(f"Timestamps: from {timestampsLeft[0]} to {timestampsLeft[-1]}")

    bonesLeft = np.array(framesLeft).transpose()
    for i in range(len(bonesLeft)):
        plt.plot(timestampsLeft, bonesLeft[i] + i * 150)
    plt.title("Bones rotation of the left hand over time in mocap file")
    plt.show()

    bonesRight = np.array(framesRight).transpose()
    for i in range(len(bonesLeft)):
        plt.plot(timestampsRight, bonesRight[i] + i * 150)
    plt.title("Bones rotation of the right hand over time in mocap file")
    plt.show()

    plt.plot(timestampsLeft)
    plt.plot(timestampsRight)
    plt.title("timestamps over frames (should be increasing linearly)")
    plt.show()

    plt.plot(posesLeft)
    plt.plot("Recognized poses of the left hand over frames")
    plt.show()

    plt.plot(posesRight)
    plt.plot("Recognized poses of the left hand over frames")
    plt.show()

    return timestampsLeft, timestampsRight, framesLeft, framesRight, posesLeft, posesRight


def checkEMG(path, mocap=None):
    """
    load a c3d file containing EMG data that can be related to motion capture
    plot the EMG next to the Motion capture to visually check if it was correctly recorded and if the synchronisation is correct
    :param path: path to the c3d file
    :param mocap: motion capture data as returned by the function checkMocap(path)
    :return: the emg data
    """
    print("EMG ------------------------------------------")
    emg = ezc3d.c3d(path)["data"]["analogs"][0]
    if len(emg[0]) == 0:  # needs recovery
        print(f"File needs recovery")
        emg = recoverFile(path)
    else:
        print("File didn't need recovery")

    print(f"{len(emg)} channels were recorded")
    print(f"{len(emg[0])} frames were recorded ({len(emg[0]) / (2000 * 60)} minutes)")

    for i in range(len(emg)):
        plt.plot(np.array(emg[i]) + i * 1500)

    if not (mocap is None):
        framesLeft = np.array(mocap[2]).transpose()
        for i in range(len(framesLeft)):
            plt.plot(np.array(mocap[0]) * 2, np.array(framesLeft[i]) * 10 + i * 1500 + 25000, c="red")
        framesRight = np.array(mocap[3]).transpose()
        for i in range(len(framesLeft)):
            plt.plot(np.array(mocap[1]) * 2, np.array(framesRight[i]) * 10 + i * 1500 + 50000, c="blue")

        plt.title("EMG channels with mocap")
    else:
        plt.title("EMG channels")
    plt.show()

    return emg


def tryClassification(dataset):
    """
    Computes simple features on the dataset containing epochs of EMG data, each associated to a single motion capture pose
    and trains a random forest to recognize the poses
    :param dataset: the epochs
    """
    print("Classification --------------------------------")
    channels = dataset.epochs().transpose((1, 0, 2))
    mav = np.array([MAV().transform(emg) for emg in channels])
    wl = np.array([WL().transform(emg) for emg in channels])
    rms = np.array([RMS().transform(emg) for emg in channels])
    maxAmp = np.array([MaximumAbsoluteAmplitude().transform(emg) for emg in channels])
    X = np.concatenate((mav, wl, rms, maxAmp)).transpose()

    print("Testing a Random Forest model (expecting good but slow results)")
    testRandomForestClass(X, dataset.poses(), "")


if __name__ == "__main__":
    arg = sys.argv
    if len(arg) < 3:
        print("You must give the path to the folder containing the data to check and the base name of the files (motion capture: [name]Quest.csv, EMG: [name].c3d")
        quit()

    foldername = arg[1]
    name = arg[2]

    pathMocap = f"{foldername}/{name}/{name}Quest.csv"
    pathEMG = f"{foldername}/{name}/{name}.c3d"

    mocap = checkMocap(pathMocap) if os.path.isfile(pathMocap) else None
    emg = checkEMG(pathEMG, mocap) if os.path.isfile(pathEMG) else None

    if not (mocap is None) and not (emg is None):
        datasetRight = EmgMocapDataSet(emg, mocap[3], mocap[1], mocap[5], windowSize=600, windowStep=300,
                                            signalCut=(30000, 30000), lowpass=150, useEnvelope=False)
        datasetLeft = EmgMocapDataSet(emg, mocap[2], mocap[0], mocap[4], windowSize=600, windowStep=300,
                                           signalCut=(30000, 30000), lowpass=150, useEnvelope=False)

        tryClassification(datasetLeft)
        tryClassification(datasetRight)
