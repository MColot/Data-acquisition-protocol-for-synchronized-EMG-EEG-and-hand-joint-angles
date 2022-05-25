"""
Experimental functions to compute the signal to noise ratio of a set of EMG channels
"""

import scipy.signal
import matplotlib.pyplot as plt
from math import log10
import numpy as np

#https://jneuroengrehab.biomedcentral.com/articles/10.1186/1743-0003-9-24
def snr1(signal):
    psd = scipy.signal.welch(signal, fs=2000, nperseg=1024)
    signalPower = sum(psd[1])
    noisePower = sum(psd[1][int(len(psd[1])/2):])/int(len(psd[1])/2)*1000
    print(10*log10(signalPower-noisePower)/noisePower)
    plt.plot(psd[1])
    plt.show()
    return 10*log10(signalPower-noisePower)/noisePower


#https://www.researchgate.net/publication/51704014_An_Algorithm_for_the_Estimation_of_the_Signal-To-Noise_Ratio_in_Surface_Myoelectric_Signals_Generated_During_Cyclic_Movements
def snr2(signal):
    r = 10
    s = 400

    epochs = np.array([[signal[i*r+j] * ((s/2)**2 - (j-s/2)**2)  for j in range(s)] for i in range(int((len(signal)-s)/r))])
    m = np.mean(epochs)

    C = np.array([sum([((s-m)**2)/r for s in e]) for e in epochs])
    C = C/np.max(C)

    log10C = [log10(e) if e > 0 else 0 for e in C]

    #plt.plot(C)
    plt.plot(log10C)
    plt.show()

    plt.hist(log10C, bins=50)
    plt.show()
    return log10C





if __name__ == "__main__":
    filename = ""

    emg = 2
    start = 30
    end = 150

    signal = np.load(filename)
    mean = 0
    for emg in range(8):
        mean += snr1(signal[emg][start * 2000:end * 2000])
    print(f"mean = {mean/8}")