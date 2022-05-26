# Data acquisition protocol for synchronized EMG EEG and hand joint angles
This work is the result of the Master thesis in computer science of Martin Colot under the supervision of Pr. Gianluca Bontempi, Pr. Guy Chéron and Cédric Simar. It was achieved at the Université Libre de Bruxelles (ULB) during the academic year 2021-2022, thanks to a collaboration with the ULB Machine Learning Group (MLG) and Laboratory of Neurophysiology and Movement Biomechanics (LNMB).

ULB: https://www.ulb.be/en

MLG: https://mlg.ulb.ac.be/wordpress

LNMB: https://www.cheron.be/lnmb_info.html

# Dataset

A dataset containing 15 subjects with synchronized EMG and hand motion capture during 2 different tasks.

Download EMG-joint angles dataset: https://drive.google.com/file/d/1pk_411pesnBrRCL7dEQdw7fxRfBiuYXa/view?usp=sharing

# Structure of the repository

## dataAcquisition
Code for the acquisition of synchronized data. The types of data that can be collected are
- motion capture of the 2 hands
- EMG of the 2 forearms
- EEG

A compiled APK file that can be loaded on an Oculus Quest to collect data using all the exercises

A simplified version of the source code of this project without libraries that cannot be shared publicly

The source python code for the remote control interface
> python dataAcquisition/remoteControl/main.py

### dataAcquisition/QuestHandTrackingLight
Unity project that can
- Runs on an Oculus Quest to record the motion capture data of the user hands and display exercices to perform.
- Display 3D animation of hands based on csv files that contain motion capture data (which can be estimated from a machine learning model).

The version of the Unity editor that was used is 2020.3.16f1

### dataAcquisition/remoteControl
Python script that can remotely send exercices and recording signal to the Oculus Quest and the EMG/EEG recirding devices.
Handles the synchronisation of the data using trigger signals and UTC.

The graphical interface can be started by the stript main.py



## dataAnalysis

A python script to perform sanity check of the collected EMG and motion capture data
> python dataAnalysis/sanityCheck.py [path to folder with all recording] [name of the session]

A python script to run classification models on a complete dataset containing multiple subects and sessions
> python dataAnalysis/mainClassificationLight.py [path to folder with all recording datasets]

A python script to run regression models on a complete dataset containing multiple subects and sessions
> python dataAnalysis/mainRegressionLight.py [path to folder with all recording datasets]

A python script to create a csv file containing finger joint angles readable by a Unity software that does visualisation of the results
> python dataAnalysis/estimationProcessing.py [path to folder with numpy files of the true gesture and the estimation] [name of the numpy file with the true gesture] [name of the numpy file with the estimation]

A Unity project to visualise the gesture estimation of a regression model


### dataAnalysis/preprocessing
Python scripts to check the acquired data (motion capture and EMG) and extract relevant information

### dataAnalysis/estimationModels
Python scripts to cut the data in epochs, extract features from it and create estimation models.
2 types of models can be created
- Classification models estimating the pose of the hand among up to 25 pre-recorded gestures
- Regression models estimating the rotation value of 16 bones of the hand at each frame

### dataAnalysis/HandGestureEstimationVisualisation
Unity project to visualise the result of the regession models

### dataAnalysis/data
Folder to place the dataset.

A complete dataset with EMG and hand motion capture can be downloaded here: https://drive.google.com/file/d/1pk_411pesnBrRCL7dEQdw7fxRfBiuYXa/view?usp=sharing


# Example videos of the result of a regression model:
Trained and tested on predefined gestures: 
https://drive.google.com/file/d/1JH87kokogjNDcUKaud-QX5BTCLn_0FgL/view?usp=sharing

Trained and tested on free gestures: 
https://drive.google.com/file/d/12d8S_mZ174Nj0abKAri2mnYk3f-vHLeS/view?usp=sharing





![ulb](https://www.ulb.be/uas/ulbout/LOGO/Logo-ULB.svg)

![mlg](https://mlg.ulb.ac.be/wordpress/wp-content/uploads/2018/05/MLG-oldQ256.png)





















