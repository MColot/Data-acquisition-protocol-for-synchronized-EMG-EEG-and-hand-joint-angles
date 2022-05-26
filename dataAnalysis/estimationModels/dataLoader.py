import numpy as np
import ezc3d
import os
from preprocessing.c3dRecovery import recoverFile

from estimationModels.dataGenerators import createDatasetLabels
from estimationModels.signalTransformation import lowpassFilter, highpassFilter
from estimationModels.timeDomainFeatures import MAV, WL, RMS, MaximumAbsoluteAmplitude
DOWNSAMPLE = 10

def loadEMG(pathEMG):
    emg = None
    recover = False
    try:
        emg = ezc3d.c3d(pathEMG)["data"]["analogs"][0]
        if len(emg[0]) == 0:  # needs recovery
            recover = True
    except:
        recover = True

    if recover:
        print(f"File needs recovery")
        emg = np.array(recoverFile(pathEMG))
    else:
        print("File didn't need recovery")
    return emg


def loadData(dataFolderPath, hands=(0, 1), datasetType=("sign", 5), labelType=0, dataType="TDF"):
    """

    :param dataFolderPath:
    :param hands:
    :param labelType: 0=pose, 1=mocap
    :param dataType: TDF=time domain features, CSP=common spatial pattern
    :return:
    """
    DATASET_NAMES = []
    for subFolder in os.listdir(dataFolderPath):
        if os.path.isdir(os.path.join(dataFolderPath, subFolder)):
            DATASET_NAMES.append(subFolder)
    X = []
    Y = []

    for name in DATASET_NAMES:
        X.append([])
        Y.append([])
        foldername = f"{dataFolderPath}/{name}"
        pathMVC = f"{foldername}/mvc/mvc.c3d"
        mvc = loadEMG(pathMVC)
        maxMVC = []
        for i in range(len(mvc)):
            maxMVC.append(max(np.abs(lowpassFilter(highpassFilter(mvc[i], 20), 150))))

        for part in range(1, datasetType[1]+1):
            pathEMG = f"{foldername}/{datasetType[0]}{part}/{datasetType[0]}{part}.c3d"
            try:
                emg = np.array([lowpassFilter(highpassFilter(channel, 20), 150) for channel in loadEMG(pathEMG)])
                for i in range(len(emg)):
                    emg[i] /= maxMVC[i]

                createDatasetLabels(dataFolderPath, name, f"{datasetType[0]}{part}")

                for s in hands:
                    side = "RL"[s]
                    print(name, part, side)

                    pathPoses = f"{dataFolderPath}/{name}/{datasetType[0]}{part}/{datasetType[0]}{part}Poses{side}.npy"
                    pathT = f"{dataFolderPath}/{name}/{datasetType[0]}{part}/{datasetType[0]}{part}Timestamps{side}.npy"
                    pathMocap = f"{dataFolderPath}/{name}/{datasetType[0]}{part}/{datasetType[0]}{part}Y{side}.npy"

                    poses = np.load(pathPoses)
                    timestamps = np.load(pathT)
                    mocap = np.load(pathMocap)
                    x = []
                    y = []
                    emgFrames = emg[s * 8:s * 8 + 8].transpose()
                    step = 2


                    for i, t in enumerate(timestamps[::step]):
                        t = int(t)
                        if labelType == 0:
                            p = poses[step * i].strip()
                            if p in ("2", "7", "19", "23") and t - 300 > 0 and 2 * t < len(emgFrames):
                                y.append(p)
                                x.append(emgFrames[t * 2 - 600:t * 2][::DOWNSAMPLE])
                        elif labelType == 1:
                            p = mocap[step * i]
                            if t - 300 > 0 and 2 * t < len(emgFrames):
                                y.append(p)
                                x.append(emgFrames[t * 2 - 600:t * 2][::DOWNSAMPLE])

                    x = np.array(x)
                    print(f"shape of x = {x.shape}")
                    if len(x) > 0:
                        if dataType == "TDF":
                            channels = x.transpose((2, 0, 1))
                            mav = np.array([MAV().transform(emg) for emg in channels])
                            wl = np.array([WL().transform(emg) for emg in channels])
                            rms = np.array([RMS().transform(emg) for emg in channels])
                            maxAmp = np.array([MaximumAbsoluteAmplitude().transform(emg) for emg in channels])
                            x = np.concatenate((mav, wl, rms, maxAmp)).transpose()
                        X[-1].append(x)
                        Y[-1].append(y)
            except Exception as e:
                print(f"Could not load file : {name}, {part}, error: {e}")

    return X, Y






def loadMocap(dataFolderPath, hands=(0, 1), datasetType=("sign", 5), labelType=0):
    """

    :param dataFolderPath:
    :param hands:
    :param labelType: 0=pose, 1=mocap
    :return:
    """
    DATASET_NAMES = []
    for subFolder in os.listdir(dataFolderPath):
        if os.path.isdir(os.path.join(dataFolderPath, subFolder)):
            DATASET_NAMES.append(subFolder)

    Y = []

    for name in DATASET_NAMES:
        Y.append([])
        foldername = f"{dataFolderPath}/{name}"


        for part in range(1, datasetType[1]+1):
            try:
                createDatasetLabels(dataFolderPath, name, f"{datasetType[0]}{part}")

                for s in hands:
                    side = "RL"[s]
                    print(name, part, side)

                    pathPoses = f"{dataFolderPath}/{name}/{datasetType[0]}{part}/{datasetType[0]}{part}Poses{side}.npy"
                    pathT = f"{dataFolderPath}/{name}/{datasetType[0]}{part}/{datasetType[0]}{part}Timestamps{side}.npy"
                    pathMocap = f"{dataFolderPath}/{name}/{datasetType[0]}{part}/{datasetType[0]}{part}Y{side}.npy"

                    poses = np.load(pathPoses)
                    timestamps = np.load(pathT)
                    mocap = np.load(pathMocap)
                    y = []
                    step = 2
                    for i, t in enumerate(timestamps[::step]):
                        t = int(t)
                        if labelType == 0:
                            p = poses[step * i].strip()
                            if p in ("2", "7", "19", "23") and t - 300 > 0:
                                y.append(p)
                        elif labelType == 1:
                            p = mocap[step * i]
                            if t - 300 > 0 :
                                y.append(p)


                    Y[-1].append(y)
            except Exception as e:
                print(f"Could not load file : {name}, {part}, error: {e}")

    return Y