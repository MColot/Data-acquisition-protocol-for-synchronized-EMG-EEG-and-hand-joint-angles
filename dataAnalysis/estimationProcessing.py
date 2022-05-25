"""
Script that uses the inverse of the PCA in order to computed the rotation values of eah bones in the 3 axis based on the single value estimated for each bone
"""

import numpy as np
import matplotlib.pyplot as plt
import sys



def bonesAngles(gesture):
    """
    returns the rotation on the 3 axis of every bone of the hand from the gesture of the pose for every frame (does the inveres of the PCA)
    """

    pca = np.load("pcaAxis.npy")
    res = []

    #add finger bones
    mean = [(30.329035415789807, -57.620789972194466), (-32.55038218541674, 1.8297571730751263),
            (-3.380466809707783, 12.147989830344994), (-4.634350571892716, 10.600418249281653),
            (2.0802188304451046, -3.3389012774220364), (1.7424163997127675, -2.520726592982014),
            (-1.7602696354966818, 3.0449899193645957), (8.274220408218609, 1.013240076268542),
            (0.6656539476232072, 1.218015460421604), (-2.243211290703704, -4.464028169857156),
            (-15.88997550960472, -6.344630460769998), (-2.7758523592969166, -2.835592530386427),
            (-3.34017571644753, 1.4300173288912577), (17.705230000007884, -5.847500000002708),
            (-12.943980365616326, -10.696538380300895), (-0.3664881118779295, -6.5106203629021),
            (-4.867913680457385, 2.919029276651357)]
    for b in range(len(pca)):  # for each bone
        res.append([])
        for i in range(len(gesture)):  # for each frame
            X = np.array([gesture[i][b], mean[b][0], mean[b][1]])
            res[-1].append(np.dot(X, pca[b]))
    return np.array(res)




if __name__ == "__main__":
    arg = sys.argv
    if len(arg) < 2:
        quit()

    pathToFolder = arg[1]
    trueGesture = arg[2]
    estimation = arg[3]


    gesture = np.load(f"{pathToFolder}/{trueGesture}.npy").transpose()
    pred = np.load(f"{pathToFolder}/{estimation}.npy").transpose()

    gestureRot = (bonesAngles(gesture).transpose((0, 2, 1)))
    predRot = (bonesAngles(pred).transpose((0, 2, 1)))

    gestureRotFrames = gestureRot.transpose((2, 0, 1))
    predRotFrames = predRot.transpose((2, 0, 1))

    csv = ""
    tot = len(gestureRotFrames)
    for i in range(10000, 20000):#len(gestureRotFrames)):
        print(f"{i}/{tot}")
        for j in range(len(gestureRotFrames[i])):
            csv += ",".join([str(e) for e in gestureRotFrames[i][j]]) + ";"
        for j in range(len(predRotFrames[i])):
            csv += ",".join([str(e) for e in predRotFrames[i][j]]) + ";"
        csv += "\n"
    with open(f"{pathToFolder}/out.csv", 'w') as f:
        f.write(csv)


