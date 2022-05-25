import struct
import numpy as np

EMG_CHANNEL_COUNT = 16


def recoverFile(fileName):
    """
    Loads a c3d file containing EMG data and extract it
    :param fileName: path to the c3d file
    :return: matrix containing each frame of the EMG for each channel
    """
    content = []
    with open(fileName, "rb") as f:
        for line in f:
            for b in line:
                content.append(b)

    startParam = content[0] * 256
    paramLen = content[startParam + 2] * 256
    analog = content[startParam + paramLen:]

    records = [analog[i * 512:(i + 1) * 512] for i in range(int(len(analog) / 512))]
    floats = []
    for r in records:
        recordFloats = [r[i * 4:(i + 1) * 4] for i in range(int(len(r) / 4))]
        b = [struct.unpack('f', bytes(f)) for f in recordFloats]
        floats.append(b)

    floats = np.concatenate(np.array(floats))

    emgs = [[] for _ in range(EMG_CHANNEL_COUNT)]
    for i in range(14464, len(floats)):
        emgs[i % EMG_CHANNEL_COUNT].append(floats[i])

    return [[e[0] for e in emg] for emg in emgs]
