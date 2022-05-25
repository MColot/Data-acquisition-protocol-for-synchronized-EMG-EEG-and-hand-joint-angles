import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import f_oneway, kruskal
from random import choices
from statsmodels.multivariate.manova import MANOVA


DATA_FOLDER_PATH = "C:/Users/marti/Desktop/memoire/data"



def analysis(data):
    print(f_oneway(*data))
    print(kruskal(*data))
    plt.boxplot(data.transpose(), showfliers=False)
    plt.title("Muscle activity vs spot number")
    plt.xlabel("spot number")
    plt.ylabel("Muscle Activity (mV)")
    plt.show()


if __name__ == "__main__":

    DATASET_NAMES = ["record_PierreC_15-9-21",
                     "record_MathieuP_16-09-21",
                     "record_Victoria_16-09-21",
                      "record_Martin_05-10-21",
                      "record_Scott_06-10-21",
                      "record_Theo_09-10-21",
                      "record_Robin_09-10-21",
                      "record_Maxime_13-10-21",
                      "record_Thomas_14-10-21",
                      "record_Cedric_14-10-21",
                      "record_Martin_15-10-21",
                      "record_MathieuB_18-10-21",
                      "record_Clara_28-10-21",
                      "record_Marie_29-10-21"]

    DATASETS_PARTS = [f"freemove{i+1}" for i in range(6)]

    fullData = []
    for name in DATASET_NAMES:
        data = []
        for part in DATASETS_PARTS:

            print(f"{name}, {part}")
            pathX = DATA_FOLDER_PATH + "/" + name + "/" + f"{part}/{part}X.npy"
            pathYL = DATA_FOLDER_PATH + "/" + name + "/" + f"{part}/{part}YL.npy"
            pathYR = DATA_FOLDER_PATH + "/" + name + "/" + f"{part}/{part}YR.npy"
            pathTL = DATA_FOLDER_PATH + "/" + name + "/" + f"{part}/{part}TimestampsL.npy"
            pathTR = DATA_FOLDER_PATH + "/" + name + "/" + f"{part}/{part}TimestampsR.npy"
            pathWristL = DATA_FOLDER_PATH + "/" + name + "/" + f"{part}/{part}WristL.npy"
            pathWristR = DATA_FOLDER_PATH + "/" + name + "/" + f"{part}/{part}WristR.npy"

            try:
                yl = np.delete(np.load(pathYL), 13, 1)  # remove bone 13 as its rotation is always constant
                yr = np.delete(np.load(pathYR), 13, 1)
                x = np.load(pathX)
                y = np.concatenate((yr, yl))
                data.extend(choices(x.transpose(), k=10000))
                #data.extend(y)
                #data.append([yl, yr])
            except Exception as e:
                print(f"{name}/{part} was not found : {e}")

        data = np.array(data)
        fullData.append(data)



    #meanData = np.array([[np.mean(sample) for sample in elem] for elem in fullData])  #subject or number of repetition
    meanData = np.array([np.concatenate([[sample[i] for sample in elem] for elem in fullData])for i in range(16)])  # channel for EMG
    #meanData = np.array([np.concatenate([np.concatenate([[s[i] for s in exo[0]] for exo in elem]) for elem in fullData]) for i in range(16)] + [np.concatenate([np.concatenate([[s[i] for s in exo[1]] for exo in elem]) for elem in fullData]) for i in range(16)])  #channel for motion capture
    analysis(meanData)




    """
    for i in range(16):
        boneData = []
        for elem in fullData:
            boneData.append(elem.transpose()[i])
        #print(f_oneway(*boneData))
        print(kruskal(*boneData))
        for b in boneData:
            plt.plot(b)



        plt.show()
        plt.hist(boneData, 30)
        plt.show()
        plt.boxplot(boneData, showfliers=False)
        plt.show()"""



