from estimationModels.utils import *
import numpy as np
from sklearn.decomposition import PCA


def extractBonesAngles(questData, side=1):
    """
    takes the content of the csv file already placed in a matrix and extract the data related to the rotation of the bones of one hand
    :param questData: matrix of the csv
    :param side: 0=left hand, 1=right hand
    :return: the rotation of the bones
    """
    res = []
    for line in questData:
        res.append([stringToVector(s) for s in line[5 + 32 * side:22 + 32 * side]])
    return np.array(res)


def extractTimeStamps(questData):
    """
    takes the content of the csv file already placed in a matrix and extract the data related to the timestamps of each frame
    :param questData: matrix of the csv
    :return: list of timestamps
    """
    res = []
    for line in questData:
        res.append(float(line[0]))
    return np.array(res)


def extractCaptureConfidence(questData, side=1):
    """
    takes the content of the csv file already placed in a matrix and extract the data indicating the confidence on the estimation of the motion capture for each finger
    :param questData: matrix of the csv
    :param side: 0=left hand, 1=right hand
    :return: list boolean indicating if all the finger are correctly captured or not
    """
    res = []
    for line in questData:
        res.append("0" not in line[26 + 32 * side:31 + 32 * side])
    return np.array(res)


def extractPose(questData, side=1):
    """
    takes the content of the csv file already placed in a matrix and extract the data related to the pose of each frame
    :param questData: matrix of the csv
    :return: list of poses
    """
    res = []
    for line in questData:
        res.append(line[32 + 32 * side])
    return np.array(res)


def extractWristRotation(questData, side=1):
    """
    takes the content of the csv file already placed in a matrix and extract the data related to the rotation of the wrist
    :param questData: matrix of the csv
    :return: list of wrist rotations
    """
    res = []
    for line in questData:
        data = line[2 + 32 * side][20:]
        # Position (-0.2503108, 0.9518678, -0.1594657), Orientation(0.7150379, 0.5020311, -0.4276603, 0.2319313)
        i = 0
        while data[i] != "(":
            i += 1

        rot = []
        i += 1
        j = i
        for c in data[i + 1:]:
            j += 1
            if c in ",)":
                rot.append(float(data[i:j]))
                i = j + 1

        res.append(rot)
    return np.array(res)


def rescaleData(data):
    """
    takes the evolution of the rotation of a bone on a single axis ond scale it between -200 and 200 degrees
    """
    res = []
    for d in data:
        while d > 200:
            d -= 360
        while d <= -200:
            d += 360
        res.append(d)
    return res


def boneSD(angles):
    """
    returns the standard deviation of the rotation of a bone on the 3 axis
    """
    mean = np.array([0.0] * 3)
    sd = np.array([0.0] * 3)
    for a in angles:
        mean += a
    mean /= len(angles)
    for a in angles:
        sd += (a - mean) ** 2
    sd /= len(angles)
    sd = sd ** 0.5
    return sd


def pcaBone(angles):
    """
    takes a list of angles of a single bone along the time and finds the axis that gives the most information on its orientation
    """
    pca = PCA(n_components=3)
    pca = pca.fit(angles)
    return [pca.components_, pca.explained_variance_ratio_]


def findBonesAxis(bones, printRes=False):
    """
    Uses a PCA to find the principal axis of a set of bones
    """
    res = []
    for i in range(17):
        s = f"{i}; "
        s += "; ".join([str(v) for v in boneSD(bones[i])]) + "; X; "

        pca = pcaBone(bones[i])
        s += "; ".join(["; ".join([str(v) for v in vect] + ["x"]) for vect in pca[0]]) + "; X; "
        s += "; ".join([str(v) for v in pca[1]]) + "; X; "

        res.append(pca[0])

        if printRes:
            print(s)
    return res


def extractMocap(pathToData, side=1, computePcaAxis = False):
    """
    reads a csv file containing motion capture data and extract relevant information (timestamps, bones rotations and poses)
    :param pathToData: path to the file containing the motion capture
    :param side: 0=left hand, 1=right hand
    :return:
    """
    pathToSavedPcaAxis = "pcaAxis.npy"

    np.set_printoptions(suppress=True)
    doc = readCsv(pathToData)
    frames = extractBonesAngles(doc, side)
    timestamps = extractTimeStamps(doc)
    poses = extractPose(doc, side)
    wrist = extractWristRotation(doc, side)
    confidences = extractCaptureConfidence(doc, side)

    # remove dirty frames
    framesClean = []
    timestampsClean = []
    posesClean = []
    wristClean = []
    for i in range(len(confidences)):
        if confidences[i]:
            framesClean.append(frames[i])
            timestampsClean.append(timestamps[i])
            posesClean.append(poses[i])
            wristClean.append(wrist[i])

    frames = np.array(framesClean)
    timestamps = np.array(timestampsClean)
    poses = np.array(posesClean)
    wrist = np.array(wristClean)
    bones = frames.transpose((1, 0, 2))

    rescaledBones = []
    for i in range(17):
        boneAxis = np.array(bones[i]).transpose()
        rescaledBoneAxis = np.array([rescaleData(boneAxis[i]) for i in range(3)])
        rescaledBones.append(rescaledBoneAxis.transpose())
    rescaledFrames = np.array(rescaledBones).transpose((1, 0, 2))

    # apply pca
    if computePcaAxis:
        pca = findBonesAxis(rescaledBones)
        # np.save(pathToSavedPcaAxis, pca)
    else:
        pca = np.load(pathToSavedPcaAxis)

    pca = np.array([p[0] for p in pca])

    processedFrames = []

    for f in rescaledFrames:
        rot = []
        for i in range(17):
            v = (f[i][0] * pca[i][0] + f[i][1] * pca[i][1] + f[i][2] * pca[i][2])
            rot.append(v)

        processedFrames.append(rot)


    return timestamps, processedFrames, poses, wrist
