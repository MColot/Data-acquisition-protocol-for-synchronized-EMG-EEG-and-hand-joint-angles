from estimationModels.dataGenerators import createDatasetLabels, createDatasetFeatures
import os
from os import path

if __name__ == "__main__":
    DATA_FOLDER_PATH = "C:/Users/marti/Desktop/memoire/data"
    NUMPY_DATA_FOLDER_PATH = "C:/Users/marti/Desktop/memoire/data/numpy"
    DATASETS_NAMES = ["record_PierreC_15-9-21",
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
                      "record_Marie_29-10-21",
                      "record_Dominique_03-11-21"]

    DATASET_PARTS_COUNT = 5
    DATASETS_PARTS = [f"sign{i + 1}" for i in range(DATASET_PARTS_COUNT)]

    for name in DATASETS_NAMES:
        for part in DATASETS_PARTS:

            if not path.exists(NUMPY_DATA_FOLDER_PATH + "/" + name):
                os.mkdir(NUMPY_DATA_FOLDER_PATH + "/" + name)

            if not path.exists(NUMPY_DATA_FOLDER_PATH + "/" + name + "/" + f"{part}"):
                os.mkdir(NUMPY_DATA_FOLDER_PATH + "/" + name + "/" + f"{part}")

            createDatasetLabels(DATA_FOLDER_PATH, name, part, NUMPY_DATA_FOLDER_PATH)
            createDatasetFeatures(DATA_FOLDER_PATH, name, part, NUMPY_DATA_FOLDER_PATH)