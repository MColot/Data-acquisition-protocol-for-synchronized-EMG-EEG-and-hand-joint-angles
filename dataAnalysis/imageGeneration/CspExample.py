import ezc3d
import numpy as np
import matplotlib.pyplot as plt

from preprocessing.c3dRecovery import recoverFile
from estimationModels.signalTransformation import Envelope
from estimationModels.dataGenerators import DataGeneratorClassification
from scipy.signal import spectrogram

from pyriemann.estimation import XdawnCovariances
from pyriemann.tangentspace import TangentSpace
from mne.decoding import CSP

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

if __name__ == "__main__":
    DATA_FOLDER_PATH = "C:/Users/marti/Desktop/memoire/data"
    DATASET_NAMES_TRAINING = ["record_Martin_05-10-21"]
    DATASET_NAMES_VALIDATION = ["record_Martin_15-10-21"]

    DATASET_PARTS_COUNT = 1
    DATASETS_PARTS = [f"sign{i + 1}" for i in range(DATASET_PARTS_COUNT)]
    generator = DataGeneratorClassification(DATA_FOLDER_PATH, DATASET_NAMES_TRAINING, 10000, parts=DATASETS_PARTS)

    data = generator.__getitem__(0, step=1)

    emg = []
    labels = []
    classesToUse = ["2", "7"]

    for i in range(len(data[0])):
        if data[1][i] in classesToUse:
            emg.append(data[0][i])
            labels.append(classesToUse.index(data[1][i]))


    emg = np.array(emg).transpose((0, 2, 1))
    labels = np.array(labels)

    csp = CSP(n_components=200, reg="oas", log=True)

    fittedCSP = csp.fit(emg, labels)
    cspX = fittedCSP.transform(emg).transpose()

    plt.scatter(cspX[0], cspX[1], c=labels)
    plt.show()




















