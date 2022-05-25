import ezc3d
import numpy as np
import matplotlib.pyplot as plt

from preprocessing.c3dRecovery import recoverFile
from estimationModels.signalTransformation import Envelope
from scipy.signal import spectrogram


def loadEMG(pathEMG):
    emg = None
    recover = False
    try:
        emg = ezc3d.c3d(pathEMG)["data"]["analogs"]#[0]
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
    emg = loadEMG("C:/Users/marti/Desktop/memoire/data/record_Martin_05-10-21/sign1/sign1.c3d")[0]

    for j in range(8):
        envelope = Envelope(emg[j], lowPass=150, sfreq=2000, highBand=20, lowBand=500)

        #plt.plot(emg[:])
        plt.plot([i for i in range(len(envelope))], envelope[:]+1000*j)

    step = 10
    #plt.plot([i*step for i in range(len(envelope)//step)], envelope[::step])
    plt.show()