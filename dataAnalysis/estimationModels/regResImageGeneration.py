import matplotlib.pyplot as plt
import numpy as np


if __name__ == "__main__":

    models = [("LR", "linear regession"), ("KNN", "KNN"), ("RF", "random forest"), ("MLP", "multi-layers perceptrons")]

    for model in models:
        actual = np.load("regRes_CSP_singleSubject/actual.npy").transpose()
        pred = np.load(f"regRes_CSP_singleSubject/pred_{model[0]}.npy").transpose()

        print(actual.shape)

        bonesToShow = (2, 4, 7, 10, 14)
        interval = [40000, 44000]

        plt.figure(figsize=(12, 7))
        for i in range(len(bonesToShow)):
            if i == 0:
                plt.plot(actual[bonesToShow[i]][interval[0]:interval[1]], color="tab:blue", label="Ground truth", linewidth=3)
                plt.plot(pred[bonesToShow[i]][interval[0]:interval[1]], color="tab:orange", label="Estimation", linewidth=1.5)
            else:
                plt.plot(actual[bonesToShow[i]][interval[0]:interval[1]] + i * 100, color="tab:blue", linewidth=3)
                plt.plot(pred[bonesToShow[i]][interval[0]:interval[1]] + i * 100, color="tab:orange", linewidth=1.5)
            plt.title(f"Joint angles estimation from a {model[1]} model")
            plt.legend(loc=1)
            plt.xlabel("Time")
            plt.ylabel("Angles")
            plt.yticks([])
            plt.xticks([])


        plt.show()