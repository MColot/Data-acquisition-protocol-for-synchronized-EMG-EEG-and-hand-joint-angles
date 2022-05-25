import math
from os import path
import os
import matplotlib.pyplot as plt
import numpy as np
from estimationModels.signalTransformation import Envelope, Normalize, epochSegmentation, lowpassFilter, highpassFilter
import ezc3d
import keras.utils.all_utils

from preprocessing.QuestDataProcessing import extractMocap
from preprocessing.c3dRecovery import recoverFile

from estimationModels.timeDomainFeatures import MAV, WL, RMS, MaximumAbsoluteAmplitude

WINDOW_SIZE = 300
MOCAP_STEP = 10
DOWNSAMPLE = 10




def createDatasetLabels(folderPath, name, part, outputFolderPath=None):
    if outputFolderPath is None:
        outputFolderPath = folderPath



    pathYL = outputFolderPath + "/" + name + "/" + f"{part}/{part}YL.npy"
    pathYR = outputFolderPath + "/" + name + "/" + f"{part}/{part}YR.npy"
    pathTL = outputFolderPath + "/" + name + "/" + f"{part}/{part}TimestampsL.npy"
    pathTR = outputFolderPath + "/" + name + "/" + f"{part}/{part}TimestampsR.npy"
    pathWristL = outputFolderPath + "/" + name + "/" + f"{part}/{part}WristL.npy"
    pathWristR = outputFolderPath + "/" + name + "/" + f"{part}/{part}WristR.npy"
    pathPosesL = outputFolderPath + "/" + name + "/" + f"{part}/{part}PosesL.npy"
    pathPosesR = outputFolderPath + "/" + name + "/" + f"{part}/{part}PosesR.npy"
    paths = pathYL, pathYR, pathTL, pathTR, pathWristL, pathWristR, pathPosesL, pathPosesL

    try:
        reload = False
        for elem in paths:
            if not path.isfile(elem):
                reload = True

        if reload:
            print("     Extracting Data")
            pathMocap = folderPath + "/" + name + "/" + f"{part}/{part}Quest.csv"
            timestampsLeft, framesLeft, posesLeft, wristLeft = extractMocap(pathMocap, 0)
            timestampsRight, framesRight, posesRight, wristRight = extractMocap(pathMocap, 1)

            np.save(pathYR, framesRight)
            np.save(pathYL, framesLeft)
            np.save(pathTR, timestampsRight)
            np.save(pathTL, timestampsLeft)
            np.save(pathWristR, wristRight)
            np.save(pathWristL, wristLeft)
            np.save(pathPosesR, posesRight)
            np.save(pathPosesL, posesLeft)
        else:
            print("     Data already extracted")
    except Exception as e:
        print(f"{name}/{part} was not found while trying to extract labels: {e}")




def createDatasetFeatures(folderPath, name, part, outputFolderPath=None):
    """
    :param folderPath:
    :param name:
    :param part:
    """
    if outputFolderPath is None:
        outputFolderPath = folderPath

    pathX = outputFolderPath + "/" + name + "/" + f"{part}/{part}X.npy"

    try:
        if not path.isfile(pathX):
            x = extractEnveloppedEMG(folderPath, name, part)
            np.save(pathX, x)
    except Exception as e:
        print(f"{name}/{part} was not found while trying to extrat emg: {e}")



def extractEnveloppedEMG(folderPath, name, part):
    """
    Load EMG information from the raw c3d files
    :param folderPath:
    :param name:
    :param part:
    :return:
    """
    pathToMVC = folderPath + "/" + name + "/" + f"mvc/mvc.c3d"
    pathEMG = folderPath + "/" + name + "/" + f"{part}/{part}.c3d"
    # emg
    def loadEMG(path):
        emg = ezc3d.c3d(path)["data"]["analogs"][0]
        if len(emg[0]) == 0:  # needs recovery
            print(f"File needs recovery")
            emg = recoverFile(path)
        return emg

    emg = loadEMG(pathEMG)
    mvc = loadEMG(pathToMVC)
    maxMVC = []
    for i in range(len(mvc)):
        maxMVC.append(max(Envelope(mvc[i], lowPass=150, sfreq=2000, highBand=20, lowBand=500)))

    envelopeData = np.array([Envelope(emg[i], lowPass=150, sfreq=2000, highBand=20, lowBand=500)[::DOWNSAMPLE] / maxMVC[i] for i in range(len(emg))])
    return envelopeData


#https://stackoverflow.com/questions/56953211/adding-input-nodes-on-an-intermediate-layer
# https://automaticaddison.com/how-to-convert-a-quaternion-into-euler-angles-in-python/
def euler_from_quaternion(x, y, z, w):
    """
    Convert a quaternion into euler angles (roll, pitch, yaw)
    roll is rotation around x in radians (counterclockwise)
    pitch is rotation around y in radians (counterclockwise)
    yaw is rotation around z in radians (counterclockwise)
    """
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll_x = math.atan2(t0, t1)

    t2 = +2.0 * (w * y - z * x)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    pitch_y = math.asin(t2)

    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw_z = math.atan2(t3, t4)

    return roll_x, pitch_y, yaw_z  # in radians


class DataGenerator(keras.utils.all_utils.Sequence):
    """
    Data loader that can return batches of data loaded from disk
    """

    def __init__(self, folderPath, datasetNames, batchSize, parts):
        self.folderPath = folderPath
        self.datasetNames = datasetNames
        self.datasetNamesAndParts = []
        self.batchSize = batchSize
        self.datasetSizes = []
        self.mvcs = dict()


        for name in self.datasetNames:
            for part in parts:
                createDatasetFeatures(self.folderPath, name, part)
                createDatasetLabels(self.folderPath, name, part)

                pathX = self.folderPath + "/" + name + "/" + f"{part}/{part}X.npy"
                pathTL = self.folderPath + "/" + name + "/" + f"{part}/{part}TimestampsL.npy"
                pathTR = self.folderPath + "/" + name + "/" + f"{part}/{part}TimestampsR.npy"

                timeStamps = (np.load(pathTL), np.load(pathTR))
                for j in range(2):
                    x = np.load(pathX)
                    i = len(timeStamps[j])-1
                    while i >= 0 and timeStamps[j][i]*2 >= len(x[0]):
                        i -= 1
                    self.datasetSizes.append(i+1)
                    self.datasetNamesAndParts.append((name, part, j))

    def __len__(self):
        return int(np.ceil(sum(self.datasetSizes)/self.batchSize))

    def setBatchSize(self, v):
        self.batchSize = v

    def findDatasets(self, idx):
        """
        returns the list of datasets to load given the id of the batch and the batch size
        :param idx: id of the batch
        :return: list of tuples of the shape (datasetName, datasetPart, (sliceStartId:sliceEndId))
        """
        cur = 0
        i = 0
        while i < len(self.datasetSizes) and cur <= idx * self.batchSize:
            cur += self.datasetSizes[i]
            i += 1
        j = i-1
        lastCount = cur - self.datasetSizes[j]
        res = [(self.datasetNamesAndParts[j][0], self.datasetNamesAndParts[j][1], self.datasetNamesAndParts[j][2], (int(idx * self.batchSize - lastCount), int(min((idx+1) * self.batchSize - lastCount, self.datasetSizes[j]))))]
        while i < len(self.datasetSizes) and cur <= (idx+1) * self.batchSize:
            intervalEnd = int(min((idx+1) * self.batchSize - cur, self.datasetSizes[i]))
            if intervalEnd > 0:
                res.append((self.datasetNamesAndParts[i][0], self.datasetNamesAndParts[i][1], self.datasetNamesAndParts[i][2], (0, intervalEnd)))
            cur += self.datasetSizes[i]
            i += 1
        return res



class DataGeneratorRegression(DataGenerator):
    def __getitem__(self, idx, step=MOCAP_STEP, predictSpeed=False):
        X = []
        Y = []
        W = []
        datasets = self.findDatasets(idx)
        for d in datasets:
            name = d[0]
            part = d[1]
            side = "LR"[d[2]]

            pathX = self.folderPath + "/" + name + "/" + f"{part}/{part}X.npy"
            pathY = self.folderPath + "/" + name + "/" + f"{part}/{part}Y{side}.npy"
            pathT = self.folderPath + "/" + name + "/" + f"{part}/{part}Timestamps{side}.npy"
            pathWristR = self.folderPath + "/" + name + "/" + f"{part}/{part}Wrist{side}.npy"

            sideForEMG = 0 if d[2] == 1 else 8
            x = np.load(pathX)[sideForEMG:8+sideForEMG].transpose()
            y = np.load(pathY)[d[3][0]:d[3][1]:step]

            # wrist = np.load(pathWristR)[d[3][0]:d[3][1]:step] # rotation as quaternion
            wrist = np.array([euler_from_quaternion(e[0], e[1], e[2], e[3]) for e in np.load(pathWristR)[d[3][0]:d[3][1]:step]]) # rotation as euler angles

            timestamps = np.load(pathT)[d[3][0]:d[3][1]:step]

            for i, t in enumerate(timestamps, start=1 if predictSpeed else 0):
                t = int(t)
                windowStart = t*2-WINDOW_SIZE*2
                if windowStart >= 0 and t*2 < len(x):
                    X.append(x[windowStart:t*2])
                    W.append(wrist[i])
                    Y.append((y[i] - y[i-1]) * (t - timestamps[i-1]) if predictSpeed else y[i])

        X = np.array(X)
        Y = np.array(Y)
        W = np.array(W)
        return [X, W], Y





class DataGeneratorClassification(DataGenerator):
    def __getitem__(self, idx, step=MOCAP_STEP, predictSpeed=False):
        X = []
        Poses = []
        datasets = self.findDatasets(idx)
        for d in datasets:
            name = d[0]
            part = d[1]
            side = "LR"[d[2]]

            print(f"{name}, {part}, {side}")

            pathX = self.folderPath + "/" + name + "/" + f"{part}/{part}X.npy"
            pathPoses = self.folderPath + "/" + name + "/" + f"{part}/{part}Poses{side}.npy"
            pathT = self.folderPath + "/" + name + "/" + f"{part}/{part}Timestamps{side}.npy"

            sideForEMG = 0 if d[2] == 1 else 1
            x = np.load(pathX)[sideForEMG*8:8+sideForEMG*8].transpose()
            poses = np.load(pathPoses)[d[3][0]:d[3][1]:step]

            timestamps = np.load(pathT)[d[3][0]:d[3][1]:step]

            XtoAdd = []

            for i, t in enumerate(timestamps, start=1 if predictSpeed else 0):
                t = int(t)
                windowStart = t*2-WINDOW_SIZE*2
                if windowStart >= 0 and t*2 < len(x):
                    XtoAdd.append(x[windowStart:t*2])
                    Poses.append((poses[i] - poses[i-1]) * (t - timestamps[i-1]) if predictSpeed else poses[i])

            """channels = np.array(XtoAdd).transpose((2, 0, 1))
            mav = np.array([MAV().transform(emg) for emg in channels])
            wl = np.array([WL().transform(emg) for emg in channels])
            rms = np.array([RMS().transform(emg) for emg in channels])
            maxAmp = np.array([MaximumAbsoluteAmplitude().transform(emg) for emg in channels])
            X.extend(np.concatenate((mav, wl, rms, maxAmp)).transpose())"""



        return [XtoAdd, np.array(Poses)]