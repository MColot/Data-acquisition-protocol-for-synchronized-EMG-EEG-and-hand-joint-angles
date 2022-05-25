import subprocess
import re
import os
from datetime import datetime
import serial
import time
import ntplib
from time import sleep, time
import msvcrt
from random import shuffle

EVENT_DURATION = 0.05  # In sec
EVENT_TO_SEND = 255
DEFAULT_EVENT_VALUE = 0

COMETA_TRIGGER_DELAY = 0.014

EMG_CHANNEL_COUNT = 16





def runBashCommand(bashCommand):
    """
	runs the given bash command
    :param bashCommand: bash command to run
    :return: the output of the command
    """
    process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
    output, error = process.communicate()
    return output, error


def runBashCommandWithDisplay(bashCommand):
    """
	run the given bash command and prints in the console the result of this command
    :param bashCommand: bash command to run
    """
    QuestOutput = runBashCommand(bashCommand)
    if QuestOutput[1] is None:
        print("	> Sucess")
    else:
        print(" > Error : {}".format(QuestOutput[1]))


def isValidFileName(s):
    """
    :param s: given file name
    :return: boolean which tells if the given file name is valid to store the collected data
    """
    if os.path.exists(s):
        print(" > you already have a recording which has that name")
        return False
    match = re.search("([a-z]*[A-Z]*[0-9]*)*", s)
    return match.group(0) == s


def recordQuest(name=None):
    """
	sends a signal to the Oculus quest to tell him to start a new recording with a given name
    """
    global currentRecording, questIsRecording, triggerTimestamp
    if name != None:
        currentRecording = name
    triggerTimestamp = None
    print("	> Sending recording signal to Quest")

    # sends a file the the Oculus quest to tell him to start its recording

    f = open("startRecording.txt", "w")
    f.write(currentRecording)
    f.close()

    pushCommand = "adb push startRecording.txt sdcard/Android/data/com.DefaultCompany.QuestHandTracking2/files/data"
    runBashCommandWithDisplay(pushCommand)

    # waits for the Oculus Quest to give feedback that its has started its recording

    commandPull = "adb pull sdcard/Android/data/com.DefaultCompany.QuestHandTracking2/files/data/startRecording.txt {}".format(
        os.getcwd().replace("\\", "/") + "/")
    done = False
    while not done:
        print("	> waiting...")
        runBashCommandWithDisplay(commandPull)
        with open("startRecording.txt", 'r') as f:
            done = f.readline() == currentRecording + " test"

    # deletes the trigger file from computer and Oculus quest

    commandDelete = "adb shell rm sdcard/Android/data/com.DefaultCompany.QuestHandTracking2/files/data/startRecording.txt"
    runBashCommandWithDisplay(commandDelete)
    os.remove("startRecording.txt")

    questIsRecording = True


def recordEMG(name=None):
    """
	sends a trigger to start the EMG recording
    """
    global EmgIsRecording, serialPort, EVENT_TO_SEND, triggerTimestamp
    if name != None:
        currentRecording = name
    print(" > Sending recording signal to EMG sensor")
    try:
        triggerTimestamp = getTime() #float(datetime.now().timestamp())
        serialPort = openSerialConnexion()
        writeOnSerialSignal(serialPort, EVENT_TO_SEND)
        EmgIsRecording = True
        print(f" > EMG recording started at {triggerTimestamp}")
    except Exception as e:
        print(f" > Failed to start recording of EMG : {e}")


def askRecordName():
    """
	asks, in the console, to give a name before starting a new recording
    """
    global currentRecording
    currentRecording = input("	> Enter record name: ")
    if not isValidFileName(currentRecording):
        print("	> Invalid Name")
        return


def startRecording():
    """
	command to start recording the EMG and the hand tracking
    """
    askRecordName()
    recordQuest()
    recordEMG()


def startRecordingQuest():
    """
	command to start recording the hand tracking only
    """
    askRecordName()
    recordQuest()


def startRecordingEMG():
    """
	command to start recording the EMG only
    """
    askRecordName()
    recordEMG()


def stopRecording():
    """
	command to stop all active recordings
	for hand tracking, sends a message to the oculus Quest to tell him to stop the recording. Then, pulls the recorded file.
	for EMG-EEG, tells the recording thread to stop itself
    """
    global currentRecording, EmgIsRecording, questIsRecording, triggerTimestamp, serialPort
    try:
        if not (questIsRecording or EmgIsRecording):
            print("	> nothing is recording")
            return
        if questIsRecording:
            print("	> Sending start recording signal to oculus Quest")
            command = "adb shell touch sdcard/Android/data/com.DefaultCompany.QuestHandTracking2/files/data/stopRecording.txt"
            runBashCommandWithDisplay(command)

            print("	> Starting download of the recording : {}".format(currentRecording))
            command = "adb pull sdcard/Android/data/com.DefaultCompany.QuestHandTracking2/files/data/{}.txt {}".format(
                currentRecording, os.getcwd().replace("\\", "/") + "/")
            runBashCommandWithDisplay(command)
        if EmgIsRecording:
            print(" > Sending stop signal to EMG sensor")
            writeOnSerialSignal(serialPort, DEFAULT_EVENT_VALUE)
            closeSerialConnexion(serialPort)
            print(" > You can save the EMG data")

        if EmgIsRecording and questIsRecording:
            pathToEMG = ""#input(" > path to EMG data for synchronisation : ")
            pathToQuest = os.getcwd().replace("\\", "/") + "/" + currentRecording + ".txt"
            synchronizeData(currentRecording, pathToEMG, pathToQuest, triggerTimestamp)

        questIsRecording = False
        EmgIsRecording = False
    except Exception as e:
        print(f"Error : {e}")


def exitProgram():
    """
	command to close to program
    """
    global exitProg
    exitProg = True
    print("	> Exiting the program")


def openSerialConnexion(comChanel='COM7'):
    """
    Opening serial port for USB2TTL8 device
    :param comChanel: COM channel to use for the serial connexion
    :return: the opened serial port
    """
    try:
        serialPort = serial.Serial(comChanel, baudrate=128000, timeout=0.01)
        serialPort.write(str.encode(f"WRITE {DEFAULT_EVENT_VALUE}\n"))
        print("Serial port opened - USB2TTL8 device LED should be green")
        return serialPort
    except Exception as e:
        print(f"No serial port opened - USB2TTL8 device not found! : {e}")


def closeSerialConnexion(serialPort):
    """
    closes the given serial port
    :param serialPort: serial port to close
    """
    try:
        serialPort.close()
    except Exception as e:
        print(f"No serial port opened! : {e}")


def writeOnSerialSignal(serialPort, value, duration=EVENT_DURATION):
    """
    writes on the given serial port, which must already be opened
    :param serialPort: serial port to write on
    :param value: value to write
    :param duration: duration during which the data must be written
    """
    global DEFAULT_EVENT_VALUE, EVENT_DURATION
    if serialPort is not None:
        serialPort.write(str.encode(f"WRITE {value}\n"))
        """
        sendingTime = time.time()
        while time.time() - sendingTime < duration:
            pass
        serialPort.write(str.encode(f"WRITE {DEFAULT_EVENT_VALUE}\n"))"""
    else:
        print(f"No serial port opened!")


def cropQuestData(pathToData, timestamp):
    """
    crops the start of a file containing hand tracking data from the Oculus quest
    so that it starts at a given timestamp
    :param pathToData: path to file to crop
    :param timestamp: timestamp that marks the start of the recording to keep
    """
    global COMETA_TRIGGER_DELAY
    resString = ""
    with open(pathToData, "r") as file:
        for line in file:
            data = line.split(";")
            dataTimestamp = float(data[0].replace(",", ".")) - COMETA_TRIGGER_DELAY
            if dataTimestamp >= timestamp:
                data[0] = str(round(1000 * (dataTimestamp - timestamp), 1)).replace(",", ".")
                resString += ";".join(data)
    with open(pathToData, "w") as file:
        file.write(resString)


def synchronizeData(name, pathToEMG, pathToQuest, startingTimestamp):
    """
    create a folder containing the recorded data from the Oculus quest and the EMG sensor
    crops the data from the Oculus Quest so that it is synchronized with the EMG data
    :param name: name of the folder to create
    :param pathToEMG: path to the file containing the EMG data
    :param pathToQuest: path to the file containing the Quest data
    :param startingTimestamp: UTC timestamp at which the recording of the EMG was started
    """
    print(f"    > synchronizing data")
    cropQuestData(pathToQuest, startingTimestamp)
    os.mkdir(name)
    #os.rename(pathToEMG, os.getcwd().replace("\\", "/") + "/" + name + "/" + currentRecording + "EMG.c3d")
    os.rename(pathToQuest, os.getcwd().replace("\\", "/") + "/" + name + "/" + currentRecording + "Quest.csv")
    print(f" > Data was saved and synchronized in folder {name}")


def reset():
    """
    send signal on quest to stop recording (even if not started) and resets everyting
    """
    global currentRecording, questIsRecording, EmgIsRecording, triggerTimestamp, serialPort

    print("	> Sending reset signal to oculus Quest")
    command = "adb shell touch sdcard/Android/data/com.DefaultCompany.QuestHandTracking2/files/data/stopRecording.txt"
    runBashCommandWithDisplay(command)

    currentRecording = ""
    questIsRecording = False
    EmgIsRecording = False
    triggerTimestamp = None
    serialPort = None

def saveGesture():
    """
    tells the oculus quest to recognize the current hand gesture and place a boolean in the data when it is done
    """
    global questIsRecording
    if questIsRecording:
        print(" > Error: recording gesture while recording is active will corrupt data")
        return
    print("	> Sending signal to Oculus Quest")
    command = "adb shell touch sdcard/Android/data/com.DefaultCompany.QuestHandTracking2/files/data/saveGesture.txt"
    runBashCommandWithDisplay(command)


def exportGestures():
    print("	> Sending signal to Oculus Quest")
    command = "adb shell touch sdcard/Android/data/com.DefaultCompany.QuestHandTracking2/files/data/exportGestures.txt"
    runBashCommandWithDisplay(command)

def importGestures():
    pushCommand = "adb push importGestures.txt sdcard/Android/data/com.DefaultCompany.QuestHandTracking2/files/data"
    runBashCommandWithDisplay(pushCommand)

def mvcInstruction():
    command = "adb shell touch sdcard/Android/data/com.DefaultCompany.QuestHandTracking2/files/data/mvc.txt"
    runBashCommandWithDisplay(command)

def signLangInstruction(rep=None, gestures=None):
    if rep is None:
        rep = input("   > Enter number of repetition: ")

    match = re.search("[0-9]*", rep)
    if match.group(0) != rep:
        print(" > Invalid number")
        return

    if gestures is None:
        gestures = input("	> Enter sequence of gesture numbers: ")
    match = re.search("([0-9]* *)*", gestures)
    if match.group(0) != gestures:
        print(" > Invalid sequence")
        return

    shuffledGestures = gestures.split(" ")
    shuffle(shuffledGestures)
    shuffledGestures = " ".join(shuffledGestures)
    print(f"shuffled gestures : {shuffledGestures}")

    f = open("signLang.txt", "w")
    f.write(rep + "\n" + shuffledGestures)
    f.close()

    pushCommand = "adb push signLang.txt sdcard/Android/data/com.DefaultCompany.QuestHandTracking2/files/data"
    runBashCommandWithDisplay(pushCommand)

    os.remove("signLang.txt")
    return f"Signs {rep} times\n{shuffledGestures}"


def freeMoveInstruction(index=None):
    if index is None:
        index = input("   > Enter exercice number (0 to 5): ")

    try:
        num = int(index)
        if num < 0 or num > 5:
            print("     Wrong number")
            return
    except:
        print("     > Wrong number")
        return

    f = open("freeMove.txt", "w")
    f.write(str(num))
    f.close()

    pushCommand = "adb push freeMove.txt sdcard/Android/data/com.DefaultCompany.QuestHandTracking2/files/data"
    runBashCommandWithDisplay(pushCommand)

    os.remove("freeMove.txt")
    return f"Freemoves {index}"


def displayUtc():
    command = "adb shell touch sdcard/Android/data/com.DefaultCompany.QuestHandTracking2/files/data/utc.txt"
    runBashCommandWithDisplay(command)


def syncUtc():
    global ntpResponse
    print("getting time")
    stop = False
    while not stop:
        try:
            print("nouvelle requête")
            c = ntplib.NTPClient()
            ntpResponse = c.request("be.pool.ntp.org", version=3)
            stop = True
            print(f"time offset from NTP : {ntpResponse.offset}")
        except:
            print("requête échouée. Attente pendant 2 secondes")
            sleep(2)

    print(f"Time sync at {getTime()}")


def getTime():
    global ntpResponse, manualTimeOffset
    try:
        return ntpResponse.offset + time() + manualTimeOffset
    except:
        return 0


def setTimeOffset(v):
    global manualTimeOffset
    manualTimeOffset = v

def getTimeOffset():
    return manualTimeOffset

exitProg = False
commands = {"utc": displayUtc, "start": startRecording, "stop": stopRecording, "startQuest": startRecordingQuest,
            "startEMG": startRecordingEMG, "reset": reset, "gesture":saveGesture, "exportGestures": exportGestures, "importGestures": importGestures, "exit": exitProgram, "mvc": mvcInstruction, "sign":signLangInstruction, "freemove":freeMoveInstruction, "sync":syncUtc}
currentRecording = ""
questIsRecording = False
EmgIsRecording = False
triggerTimestamp = None
serialPort = None

ntpResponse = 0
manualTimeOffset = 0

if __name__ == "__main__":


    print("Don't forget to synchronize before recording simultaneous Hand tracking and EMG/EEG")

    while not exitProg:
        print("------------------------------")
        print("Interface for the EMG-Hand Tracking synchronization system")
        print("Commands:")
        print("utc: display UTC time stamps")
        print("sync: gets utc from npt server to synchronize with quest")
        print("start: start recording")
        print("stop: stop recording")
        print("startQuest: start recording of quest only")
        print("startEMG: start recording of EMG only")
        print("mvc: tells the Oculus quest to display the instructions for mvc")
        print("sign: tells the Oculus quest to display the instructions for sign language")
        print("freemove: tells the Oculus quest to display the instructions for free moves")
        print("reset: stops recording on the quest and resets everything")
        print("gesture: tells the Oculus quest to recognize the current hand gesture")
        print("exportGestures: export the currently recorded gestures in a text file on the oculus quest")
        print("importGestures: import the gestures in the file 'importGestures.txt' on the oculus quest")
        print("exit: exit the program")
        print("------------------------------")

        inputCommand = input("Enter command: ")
        if inputCommand in commands:
            commands[inputCommand]()
        else:
            print("	> Wrong command")
        print("")


"""
This scripts synchrinizes the quest data with the EMG data by cropping the quest data
The EEG data must be cropped by taking into account the delay for its trigger and the delay for the trigger of the EMG


mouvements 0, 13, 16 et 18 envoient au casque le signal d'ouvrir le menu
"""