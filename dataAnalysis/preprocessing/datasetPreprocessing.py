import numpy as np
import os, sys
import ezc3d
from preprocessing.c3dRecovery import recoverFile
from preprocessing.QuestDataProcessing import extractMocap
import matplotlib.pyplot as plt


def preprocess(datasetPath):
    """
    Takes a directory and extract all the EMG and motion capture data in the file that are in this directory
    :param datasetPath: path to the directory
    """
    outPath = f"{datasetPath}/datasets"
    if not os.path.exists(outPath):
        os.mkdir(outPath)

    for (dirPath, dirNames, fileNames) in os.walk(datasetPath):
        for fileName in fileNames:
            # EMG extraction
            filePath = f"{dirPath}/{fileName}".replace("\\", "/")
            parentDirName = os.path.dirname(filePath).split('/')[-1]
            if fileName[-4:] == ".c3d":
                print(f"Reading file {fileName}")
                emg = ezc3d.c3d(filePath)["data"]["analogs"][0]
                if len(emg[0]) == 0:  # needs recovery
                    print(f"Recovering file {fileName}")
                    emg = recoverFile(filePath)
                np.save(f"{outPath}/{parentDirName}_EMG.npy".replace("\\", "/"), emg)
            # mocap extraction
            elif fileName[-4:] == ".csv":
                print(f"Reading file {fileName}")
                timestamps, frames, poses = extractMocap(filePath)
                np.save(f"{outPath}/{parentDirName}_mocapFrames.npy", frames)
                np.save(f"{outPath}/{parentDirName}_mocapTimestamps.npy", timestamps)
                np.save(f"{outPath}/{parentDirName}_mocapPoses.npy", poses)


if __name__ == "__main__":
    arg = sys.argv
    if len(arg) == 1:
        # quit()
        arg = (arg[0], "C:/Users/marti/Desktop/stage/recordings/Antoine_22-07-21/antoineFreeMove1")
    datasetPath = arg[1]
    preprocess(datasetPath)
